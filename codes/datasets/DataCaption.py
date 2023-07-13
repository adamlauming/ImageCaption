'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-05 22:25:41
'''
import os
import sys
import random
import numpy as np
import torch
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import jieba
import json
from string import punctuation
import re

import datasets.utils_data as utils

#===================================================================================
# caption dataset - 1 show attend and tell !!!
#===================================================================================
class DatasetCaption(Dataset):
    def __init__(self, data_dir, Flags, mode='train'):
        super().__init__()
        self.mode = mode
        self.Flags = Flags
        self.inputsize = [Flags.img_size, Flags.img_size]

        self.image_dir = os.path.join(data_dir, '{}data'.format(mode))
        self.label_dir = os.path.join(data_dir, '{}labels.csv'.format(mode))
        self.listfile = os.path.join(data_dir, '{}files.txt'.format(mode))

        self.filenames = utils.txt2list(self.listfile)
        print("Num of {} images:  {}".format(mode, len(self.filenames)))

        self.to_tensor = transforms.ToTensor()

        # lablefile
        with open(self.label_dir, "r") as f:
            reader = csv.reader(f)
            self.label_file = list(reader)

        # caption
        self.word_map_dir = os.path.join(data_dir, 'WORDMAP_unstructed.json')
        with open(self.word_map_dir, 'r', encoding='utf-8') as f:
            self.word_map = json.load(f)
        self.punc = punctuation + u'um 1234567890'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        batch_name = self.filenames[index]  # only 1 image name return
        image_arr1, image_arr2 = self.get_img(batch_name, mode=self.mode)
        x_data1 = self.to_tensor(image_arr1.copy()).float()
        x_data2 = self.to_tensor(image_arr2.copy()).float()

        for row in self.label_file:
            if batch_name in row:
                label = torch.tensor(list(map(int, row[1:12]))).float()
                diagnose = row[12]
                diagnose = re.sub(r"[{}]+".format(self.punc), "", diagnose)
                diagnose = list(jieba.cut(diagnose))
                enc_diagnose = utils.encode_caption(self.word_map, diagnose)
                caption = torch.LongTensor(enc_diagnose)
                caplen = torch.LongTensor([len(diagnose) + 2])
                break

        return x_data1, x_data2, label, caption, caplen, batch_name

    # load images and labels depend on filenames
    def get_img(self, file_name, mode='train'):
        if 'train' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im1 = utils.random_perturbation(image_im1)
            image_im1 = utils.random_geometric3(image_im1)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)
            image_im2 = utils.random_perturbation(image_im2)
            image_im2 = utils.random_geometric3(image_im2)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'val' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'test' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2
        

#===================================================================================
# caption dataset - 2 hierarchical RNN !!!
#===================================================================================
class DatasetCaption2(Dataset):
    def __init__(self, data_dir, Flags, mode='train'):
        super().__init__()
        self.mode = mode
        self.Flags = Flags
        self.inputsize = [Flags.img_size, Flags.img_size]

        self.image_dir = os.path.join(data_dir, '{}data'.format(mode))
        self.label_dir = os.path.join(data_dir, '{}labels.csv'.format(mode))
        self.listfile = os.path.join(data_dir, '{}files.txt'.format(mode))

        self.filenames = utils.txt2list(self.listfile)
        print("Num of {} images:  {}".format(mode, len(self.filenames)))

        self.to_tensor = transforms.ToTensor()

        # lablefile
        with open(self.label_dir, "r") as f:
            reader = csv.reader(f)
            self.label_file = list(reader)

        # caption
        self.word_map_dir = os.path.join(data_dir, 'WORDMAP_unstructed.json')
        with open(self.word_map_dir, 'r', encoding='utf-8') as f:
            self.word_map = json.load(f)
        self.punc = punctuation + u'um 1234567890'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        batch_name = self.filenames[index]  # only 1 image name return
        image_arr1, image_arr2 = self.get_img(batch_name, mode=self.mode)
        x_data1 = self.to_tensor(image_arr1.copy()).float()
        x_data2 = self.to_tensor(image_arr2.copy()).float()

        for row in self.label_file:
            if batch_name in row:
                label = list(map(int, row[1:12]))
                diagnose = row[12]
                diagnose = re.sub(r"[{}]+".format(self.punc), "", diagnose)

                target_sentence = []
                caplen = 0 # the length of the longest sentence in each diagnose
                for i, sentence in enumerate(diagnose.split('。')[:-1]):
                    tokens = []
                    sentence_cut = jieba.cut(sentence)
                    tokens.append(self.word_map['<start>'])
                    tokens.extend([self.word_map[word] if word in self.word_map.keys() else self.word_map['<unk>'] for word in sentence_cut])        
                    tokens.append(self.word_map['<end>'])            
                    target_sentence.append(tokens)
                    if caplen < len(tokens):
                        caplen = len(tokens)
                sennum = len(target_sentence)
                break

        return x_data1, x_data2, label, target_sentence, caplen, sennum, batch_name

    # load images and labels depend on filenames
    def get_img(self, file_name, mode='train'):
        if 'train' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im1 = utils.random_perturbation(image_im1)
            image_im1 = utils.random_geometric3(image_im1)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)
            image_im2 = utils.random_perturbation(image_im2)
            image_im2 = utils.random_geometric3(image_im2)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'val' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

        elif 'test' in self.mode:
            image_file_dir = os.path.join(self.image_dir, file_name)
            image_files = utils.all_files_under(image_file_dir, append_path=False)
            if len(image_files) < 2:
                print(image_file_dir)
            images_file_1 = os.path.join(image_file_dir, image_files[0])
            images_file_2 = os.path.join(image_file_dir, image_files[1])
            image_im1 = Image.open(images_file_1).resize(self.inputsize)
            image_im2 = Image.open(images_file_2).resize(self.inputsize)

            image_arr1 = np.array(image_im1, dtype=np.float32) / 255.0
            image_arr2 = np.array(image_im2, dtype=np.float32) / 255.0

            return image_arr1, image_arr2

def my_collate_fn(data):
    x_data1, x_data2, label, target_sentence, caplen, sennum, batch_name = zip(*data)
    x_data1 = torch.stack(x_data1)
    x_data2 = torch.stack(x_data2)

    max_sentence_num = max(sennum)
    max_word_num = max(caplen)

    word_num_gts = np.zeros((len(target_sentence), max_sentence_num))
    for i, captions in enumerate(target_sentence):
        for j, sentence in enumerate(captions):
            word_num_gts[i][j] = len(sentence)

    targets = np.zeros((len(target_sentence), max_sentence_num, max_word_num))
    prob = np.zeros((len(target_sentence), max_sentence_num))

    # 将输出batch转化为列表
    for i, captions in enumerate(target_sentence):
        for j, sentence in enumerate(captions):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0
    
    label = torch.Tensor(label).float()
    return x_data1, x_data2, label, torch.from_numpy(targets).type(torch.LongTensor), torch.from_numpy(prob).type(torch.LongTensor), torch.from_numpy(word_num_gts).type(torch.LongTensor), batch_name


