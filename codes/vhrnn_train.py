'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-15 12:55:01
'''
import argparse
import os
import sys
import jieba
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

from datasets.DataCaption import *
from models.Losses import *
from models.network_b import *
from models.network_v import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
from nltk.translate.bleu_score import corpus_bleu

#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=10, type=int, help="check free gpu interval")
parser.add_argument('--log_cols', default=140, type=int, help="num of columns for log")

parser.add_argument('--epochs', default=80, type=int, help="nums of epoch")
parser.add_argument('--inchannels', default=3, type=int, help="nums of input channels")
parser.add_argument('--img_size', default=320, type=int, help="image size")
parser.add_argument('--n_class', default=11, type=int, help="output classes")
parser.add_argument('--label_names', default=['MH','SP','ED','MEM','PVD','ILM','PD','IO','RNL','RPE','CA'], nargs='+', help="output class names")
parser.add_argument('--datatrain', default='train', type=str, help="select all data or part data train")
parser.add_argument('--batch_size', default=8, type=int, help="batch size")
parser.add_argument('--workers', default=4, type=int, help="num of workers")
parser.add_argument('--model', default='Caption2', type=str, help="training model")
parser.add_argument('--encoder', default='resnet50', type=str, help="training model")
parser.add_argument('--en_pretrained', default=True, type=bool, help="whether load pretrained model")
parser.add_argument('--learning_rate', default=2e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--alpha', default=2, type=float, help="focal loss weigth")
parser.add_argument('--gamma', default=2, type=float, help="focal loss param")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")
parser.add_argument('--word_map_len', default=5, type=int, help="word_map_len")
parser.add_argument('--savename', default='Result', type=str, help="output folder name")
parser.add_argument('--lambda_stop', default=3, type=float, help="stop loss param")
parser.add_argument('--lambda_word', default=1, type=float, help="word loss param")
parser.add_argument('--lambda_cla', default=3, type=float, help="cla loss param")

Flags, _ = parser.parse_known_args()
utils.ShowFlags(Flags)
os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(Flags.gpu_gap, Flags.gpu_rate)
torch.cuda.empty_cache()

#==============================================================================
# Dataset
#==============================================================================
data_dir = os.path.join('..', '..', 'OCTMultiCLA/data')

jieba.load_userdict(os.path.join(data_dir, 'user_dict.txt')) # dictionary
word_map_file = os.path.join(data_dir, 'WORDMAP_unstructed.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)       # Read word map
Flags.word_map_len = len(word_map)
start_token = word_map['<start>']


dataset_train = DatasetCaption2(data_dir, Flags, mode=Flags.datatrain)
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True, collate_fn=my_collate_fn)

dataset_val = DatasetCaption2(data_dir, Flags, mode=Flags.datatrain.replace('train', 'val'))
dataloader_val = DataLoader(dataset_val, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True, collate_fn=my_collate_fn)

utils.log('Load Data successfully')

#==============================================================================
# Logger
#==============================================================================
logger = Logger(Flags)
utils.log('Setup Logger successfully')

#==============================================================================
# load model, optimizer, Losses
#==============================================================================
model = globals()[Flags.model](Flags)
model = model.cuda() if torch.cuda.is_available() else model
print('load model {}'.format(Flags.model))
# summary(model, (3, 512, 512))
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': BCELoss(),
    'CE': nn.CrossEntropyLoss(),
    'DICE': DiceLoss(),
    'FOCAL': FocalLoss(gamma=Flags.gamma, alpha=[Flags.alpha, 1]),
}
metrics = CaptionMetrics()
utils.log('Build Model successfully')

#==============================================================================
# Train model
#==============================================================================
for epoch in range(Flags.epochs + 1):
    ############################################################
    # Train Period
    ############################################################
    model.train()
    pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
    pbar.set_description('Epoch {:2d}'.format(epoch))

    log_Train, temp = {}, {}
    all_true, all_pred = [], []
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch1 = batch_data[0]
        x_batch2 = batch_data[1]
        y_batch = batch_data[2]
        caption_batch = batch_data[3]
        prob_batch = batch_data[4]
        caplen_batch = batch_data[5]
        batch_name = batch_data[6]

        x_data1 = utils._to_var(x_batch1)
        x_data2 = utils._to_var(x_batch2)
        context = utils._to_var(torch.LongTensor(caption_batch).long(), requires_grad=False)
        prob_real = utils._to_var(torch.LongTensor(prob_batch).float(), requires_grad=False)
        y_true = y_batch.cuda()
        optimizer.zero_grad()

        # forward
        outputs = model(x_data1, x_data2, context)

        y_pred = outputs["classify_out"]
        y_pred = torch.sigmoid(y_pred)
        loss_cla = criterion['BCE'](y_true, y_pred)

        stop_pred = outputs["stop_out"]
        loss_stop, loss_word = 0, 0
        for i in range(len(stop_pred)):
            stop_pred_s = torch.softmax(stop_pred[i], dim=1)
            loss_stop += criterion['BCE'](prob_real[:,i], stop_pred_s[:,1])
        
        words = outputs["word_out"]
        for sentence_index in range(context.shape[1]):
            sentence_pred = words[sentence_index]
            for word_index in range(context.shape[2] - 1):
                word_mask = (context[:, sentence_index, word_index+1] > 0).float()
                word_pred = sentence_pred[word_index]
                loss_word += (criterion['CE'](word_pred, context[:, sentence_index, word_index + 1]) * word_mask).sum()

        # backward
        loss = Flags.lambda_stop * loss_stop + Flags.lambda_word * loss_word + Flags.lambda_cla * loss_cla
        loss.backward()
        optimizer.step()

        # metric
        y_true = y_true.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)

        # log
        temp['Loss'] = loss.item()
        log_Train = utils.MergeLog(log_Train, temp, n_step)
        pbar.set_postfix(log_Train)
    # cap_true, cap_pred = np.hstack(cap_true), np.hstack(cap_pred)
    # acc, _ = metrics.ClassifyScore(cap_true, cap_pred)
    # log_Train['Acc']= acc
    # pbar.set_postfix(log_Train)
    logger.write_tensorboard('1.Train', log_Train, epoch)

    if not (epoch % Flags.val_step == 0):
        continue

    ############################################################
    # Test Period
    ############################################################
    print('*' * Flags.log_cols)
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader_val, ncols=Flags.log_cols)
        pbar.set_description('Val')
        log_Test, temp = {}, {}
        all_true, all_pred = [], []
        references, hypotheses = [], []
        for n_step, batch_data in enumerate(pbar):
             # get data
            x_batch1 = batch_data[0]
            x_batch2 = batch_data[1]
            y_batch = batch_data[2]
            caption_batch = batch_data[3]
            prob_batch = batch_data[4]
            caplen_batch = batch_data[5]
            batch_name = batch_data[6]

            x_data1 = utils._to_var(x_batch1)
            x_data2 = utils._to_var(x_batch2)
            context = utils._to_var(torch.LongTensor(caption_batch).long(), requires_grad=False)
            prob_real = utils._to_var(torch.LongTensor(prob_batch).float(), requires_grad=False)
            y_true = y_batch.cuda()

            # forward
            outputs = model(x_data1, x_data2, context)

            y_pred = outputs["classify_out"]
            y_pred = torch.sigmoid(y_pred)
            loss_cla = criterion['BCE'](y_true, y_pred)

            stop_pred = outputs["stop_out"]
            loss_stop, loss_word = 0, 0
            for i in range(len(stop_pred)):
                stop_pred_s = torch.softmax(stop_pred[i], dim=1)
                loss_stop += criterion['BCE'](prob_real[:,i], stop_pred_s[:,1])
            
            words = outputs["word_out"]
            predicts_sentences = np.zeros((context.shape[0], context.shape[1], context.shape[2]))
            for sentence_index in range(context.shape[1]):
                sentence_pred = words[sentence_index]
                for word_index in range(context.shape[2] - 1):
                    word_mask = (context[:, sentence_index, word_index+1] > 0).float()
                    word_pred = sentence_pred[word_index]
                    loss_word += (criterion['CE'](word_pred, context[:, sentence_index, word_index + 1]) * word_mask).sum()
                    pred_word = word_pred.max(1)[1].cpu()
                    for bs in range(context.shape[0]):
                        if word_mask[bs] > 0:
                            predicts_sentences[bs, sentence_index, word_index] = pred_word[bs]

            loss = Flags.lambda_stop * loss_stop + Flags.lambda_word * loss_word + Flags.lambda_cla * loss_cla
            temp['Loss'] = loss.item()

            # metric
            y_true = y_true.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            all_true.append(y_true)
            all_pred.append(y_pred)

            log_Test = utils.MergeLog(log_Test, temp, n_step)
            pbar.set_postfix(log_Test)

            # sample words
            for idx_pred in range(len(predicts_sentences)):
                pred_sentence = []
                for seq_idx in range(len(predicts_sentences[idx_pred])):
                    for word_idx in range(len(predicts_sentences[idx_pred][seq_idx])):
                        if predicts_sentences[idx_pred][seq_idx][word_idx] not in [word_map['<start>'], word_map['<pad>']]:
                            pred_sentence.append(predicts_sentences[idx_pred][seq_idx][word_idx])
                hypotheses.append(pred_sentence)

            captions = caption_batch.numpy()
            for idx_ref in range(len(captions)):
                ref_sentence = list()
                for seq_idx in range(len(captions[idx_ref])):
                    for word_idx in range(len(captions[idx_ref][seq_idx])):
                        if captions[idx_ref][seq_idx][word_idx] not in [word_map['<start>'], word_map['<pad>']]:
                            ref_sentence.append(captions[idx_ref][seq_idx][word_idx])
                references.append(ref_sentence)

            assert len(references) == len(hypotheses)

        # log
        bleu4 = corpus_bleu(np.expand_dims(references, axis=1), hypotheses)

        log_Test['Loss'] = loss.item()
        log_Test['BLEU'] = bleu4
        print(log_Test)
        logger.write_tensorboard('2.Val', log_Test, epoch)
        logger.save_model(model, 'Ep{}_BLEU_{:.4f}'.format(epoch, bleu4))
    print('*' * Flags.log_cols)