'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-04-13 19:33:13
'''
import torch
import torch.nn as nn
import os
import json

min_word_feq = 3
train_filename = './codes/data/2019-all/traindata_unstructed.txt'
val_filename = './codes/data/2019-all/valdata_unstructed.txt'
test_filename = './codes/data/2019-all/testdata_unstructed.txt'

image_folder = './2019'
data_folder = './codes/data/2019-all'

word_map_file = os.path.join(data_folder, 'WORDMAP_unstructed.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)
        
weights_final = [0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.007042253521126761, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.007042253521126761, 0.007042253521126761, 0.007042253521126761, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.007042253521126761, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804, 0.0035211267605633804]
weights_unstructed = [0.003401360544217687,0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.006802721088435374, 0.006802721088435374, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687, 0.003401360544217687]

max_len = 70
batch_size = 24

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 1e-4  # learning rate for decoder
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 30  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
# checkpoint = 'checkpoint_.pth.tar'
best_checkpoint = '/home/wt/caption/show_attend_and_tell/show_attend_and_tell_3/save_model/0322_resnet50_notf_unstructed_weight/BEST_checkpoint_.pth.tar'
workers = 2
grad_clip = 5.  # clip gradients at an absolute value of

bce = nn.BCELoss().to(device)


















