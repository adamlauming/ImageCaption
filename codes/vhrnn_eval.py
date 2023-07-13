'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-16 13:41:38
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
from models.network_b import *
from models.network_v import *
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
from collections import Counter

utils.log('start evaluation')

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='T1030-1739-Caption2-Lstop3-Lword1-Lcla3', type=str, help="get center type")
parser.add_argument('--img_size', default=320, type=int, help="image size")
parser.add_argument('--mode', default='val', type=str, help="get center type")
parser.add_argument('--network', default='Caption2', type=str, help="training model")
parser.add_argument('--encoder', default='resnet50', type=str, help="training model")
parser.add_argument('--word_map_len', default=5, type=int, help="word_map_len")
parser.add_argument('--label_names', default=['MH','SP','ED','MEM','PVD','ILM','PD','IO','RNL','RPE','CA'], nargs='+', help="output class names")

Flags, _ = parser.parse_known_args()
# load model
pypath = os.path.abspath(__file__)
path, _ = os.path.split(pypath)
weightname = os.path.join(path, '..', Flags.model, 'Model', 'Model_Ep50_BLEU_0.5084.pkl')
model = torch.load(weightname)
model.eval()
metrics = CaptionMetrics()

# load data
data_dir = os.path.join('..', '..', 'OCTMultiCLA/data')
dataset_val = DatasetCaption2(data_dir, Flags, mode=Flags.mode)
dataloader_val = DataLoader(dataset_val, batch_size=1, collate_fn=my_collate_fn)

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
end_token = word_map['<end>']

# load word2id
with open(os.path.join(data_dir, 'word2idx_unstructed.json'), 'r', encoding='utf-8') as j:
    id2_word = json.load(j)

#%% Evaluation
pbar = tqdm(dataloader_val, ncols=60)
all_references, all_hypotheses = [], []
all_true, all_pred = [], []
with torch.no_grad():
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch1 = batch_data[0]
        x_batch2 = batch_data[1]
        y_batch = batch_data[2]
        caption_batch = batch_data[3]
        prob_batch = batch_data[4]
        caplen_batch = batch_data[5]
        data_name = batch_data[6]

        x_data1 = utils._to_var(x_batch1)
        x_data2 = utils._to_var(x_batch2)
        context = utils._to_var(torch.LongTensor(caption_batch).long(), requires_grad=False)
        prob_real = utils._to_var(torch.LongTensor(prob_batch).float(), requires_grad=False)
        y_true = y_batch.cuda()

        # forward
        outputs = model.sample(x_data1, x_data2, start_token)

        y_pred = outputs["classify_out"]
        y_pred = torch.sigmoid(y_pred)

        stop_pred = outputs["stop_out"]
        words = outputs["word_out"]
        p_stop_sentences = torch.zeros(10)
        predicts = []
        for sentence_index in range(10):
            stop_pred[sentence_index] = torch.max(stop_pred[sentence_index], 1)[1]
            p_stop_sentences[sentence_index] = stop_pred[sentence_index]
            words[sentence_index] = words[sentence_index] * stop_pred[sentence_index]
            predicts.append(words[sentence_index].squeeze().cpu().numpy())

        captions = np.array(caption_batch).squeeze(axis=0)
        
        hypotheses, references = [], []
        hypotheses.append([predicts[j][i] for j in range(len(predicts)) for i in range(len(predicts[j])) if predicts[j][i] not in [word_map['<start>'], word_map['<pad>']]])
        references.append([captions[j][i] for j in range(len(captions)) for i in range(len(captions[j])) if captions[j][i] not in [word_map['<start>'], word_map['<pad>']]])
        
        img_captions_words = []
        preds_words_repeat = []
        for i in range(len(references)):
            img_captions_word = ''
            preds_word = ''
            for j in references[i]:
                word = id2_word[str(j)]
                if word == '<start>': continue
                if word == '<pad>': break
                if word == '<end>': word = '。'
                img_captions_word += word
            for l in hypotheses[i]:
                word = id2_word[str(l)]
                if word == '<start>': continue
                if word == '<pad>': break
                if word == '<end>': word = '。'
                preds_word += word
            img_captions_words.append(img_captions_word)
            preds_words_repeat.append(preds_word)  

        preds_words = []
        for sentence in preds_words_repeat:
            sen_counter = Counter()
            sentence_split = sentence.split('。')[:-1]
            sen_counter.update(sentence_split)
            seq_no_repeat = ''
            for k, v in sen_counter.items():
                seq_no_repeat += (k+'。')
            preds_words.append(seq_no_repeat)
            sentence_cut = jieba.cut(seq_no_repeat)
            all_hypotheses.append([word_map[word] if word in word_map.keys() else word_map['<unk>'] for word in sentence_cut])

        for sentence in img_captions_words:
            sentence_cut = jieba.cut(sentence)
            all_references.append([word_map[word] if word in word_map.keys() else word_map['<unk>'] for word in sentence_cut])

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