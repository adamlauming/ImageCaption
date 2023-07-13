'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-05 22:24:02
'''
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import jieba

from datasets.DataCaption import *
from models.Losses import *
from models.network_a import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
import pandas as pd
from sklearn.metrics import *
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.bleu_score import corpus_bleu

utils.log('start evaluation')

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='T1030-1044-Caption1', type=str, help="get center type")
parser.add_argument('--img_size', default=320, type=int, help="image size")
parser.add_argument('--mode', default='val', type=str, help="get center type")
parser.add_argument('--network', default='Caption1', type=str, help="training model")
parser.add_argument('--encoder', default='resnet50', type=str, help="training model")
parser.add_argument('--word_map_len', default=5, type=int, help="word_map_len")
parser.add_argument('--label_names', default=['MH','SP','ED','MEM','PVD','ILM','PD','IO','RNL','RPE','CA'], nargs='+', help="output class names")

Flags, _ = parser.parse_known_args()
# load model
pypath = os.path.abspath(__file__)
path, _ = os.path.split(pypath)
weightname = os.path.join(path, '..', Flags.model, 'Model', 'Model_Ep50_BLEU_0.4132.pkl')
model = torch.load(weightname)
model.eval()
metrics = CaptionMetrics()

# load data
data_dir = os.path.join('..', '..', 'OCTMultiCLA/data')
dataset_val = DatasetCaption(data_dir, Flags, mode=Flags.mode)
dataloader_val = DataLoader(dataset_val, batch_size=6)

# save config
_, name = os.path.split(weightname)
name, _ = os.path.splitext(name)
save_dir = os.path.join(path, '..', 'ModelResults', Flags.network + name)
print(save_dir)
utils.checkpath(save_dir)

pred_dir = os.path.join(save_dir, 'Captions.csv')
with open(pred_dir, "a") as f:
    w = csv.writer(f)
    w.writerow(['ID']+['Captions'])

# Load word map (word2ix)
jieba.load_userdict(os.path.join(data_dir, 'user_dict.txt')) # dictionary
word_map_file = os.path.join(data_dir, 'WORDMAP_unstructed.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)
Flags.word_map_len = len(word_map)
start_token = word_map['<start>']

# load word2id
with open(os.path.join(data_dir, 'word2idx_unstructed.json'), 'r', encoding='utf-8') as j:
    id2_word = json.load(j)

#%% Evaluation
pbar = tqdm(dataloader_val, ncols=60)
all_references = list()
all_hypotheses = list()
all_true, all_pred = [], []
with torch.no_grad():
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch1 = batch_data[0]
        x_batch2 = batch_data[1]
        y_batch = batch_data[2]
        caption_batch = batch_data[3]
        caplen_batch = batch_data[4]
        data_name = batch_data[5]
        if torch.cuda.is_available():
            x_data1 = x_batch1.cuda()
            x_data2 = x_batch2.cuda()
            y_true = y_batch.cuda()
            caption_true = caption_batch.cuda()
            caplen = caplen_batch.cuda()

        outputs = model.sample_t(x_data1, x_data2, start_token)
        y_pred = outputs["classify_out"]
        y_pred = torch.sigmoid(y_pred)

        scores = outputs["main_out"]
       
        # metric
        y_true = y_true.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)

        references, hypotheses = [], []
        for j in range(caption_true.shape[0]):
            img_caps = caption_true[j].tolist()
            img_caption = []
            for w in img_caps:
                if w not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}:
                    img_caption.append(w)  # remove <start> and pads
            references.append(img_caption)
            all_references.append(img_caption)

        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            for w in preds[j]:
                if w in {word_map['<end>']}: w_index = preds[j].index(w)
            temp_preds.append(preds[j][:w_index])
        preds = temp_preds
        hypotheses.extend(preds)
        all_hypotheses.extend(preds)
        # print(hypotheses)
        assert len(references) == len(hypotheses)
        
        img_captions_words = []
        preds_words = []
        for i in range(len(references)):
            img_captions_word = ''
            preds_word = ''
            for j in references[i]:
                # img_captions_word.append(id2_word[str(j)])
                img_captions_word += id2_word[str(j)]
            for l in hypotheses[i]:
                # preds_word.append(id2_word[str(l)])
                preds_word += id2_word[str(l)]
            img_captions_words.append(img_captions_word)
            preds_words.append(preds_word)

        for batchidx in range(x_data1.shape[0]):
            name_, _ = os.path.splitext(data_name[batchidx])
            with open(pred_dir, "a") as f:
                w = csv.writer(f)
                rows1 = [name_] + [img_captions_words[batchidx]]
                rows2 = [name_] + [preds_words[batchidx]]               
                w.writerow(rows1)
                w.writerow(rows2)

    # metric
    bleu4 = corpus_bleu(np.expand_dims(all_references, axis=1), all_hypotheses)
    print('bleu4 =', bleu4)

    gts = dict()
    res = dict()
    for i in range(len(all_hypotheses)):
        gts[str(i)] = [' '.join(str(w) for w in all_hypotheses[i])]
        res[str(i)] = [' '.join(str(w) for w in all_references[i])]
    bleu_score = Bleu()
    bleu_scorer, bleu_scorers = bleu_score.compute_score(res, gts)
    cide_score = Cider()
    cider_score, cider_scores = cide_score.compute_score(res, gts)
    rouge_score = Rouge()
    rouge_scorer, rouge_scorers = rouge_score.compute_score(res, gts)
    meteor_score = Meteor()
    meteor_scorer, mouge_scorers = meteor_score.compute_score(res, gts)

    print('bleu_scorer =', bleu_scorer)
    print('cider =', cider_score)
    print('rouge =', rouge_scorer)
    print('meteor =', meteor_scorer)

