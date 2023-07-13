'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-06 14:10:10
'''
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchsummary
from torch.nn import functional as F
from torch.nn import init
from torchvision import models


from models.layers.init_weights import init_weights
from models.layers.blocks import *
import segmentation_models_pytorch as smp
import models.resnet as resnet

data_dir = os.path.join('..', '..', 'OCTMultiCLA/data')
word_map_file = os.path.join(data_dir, 'WORDMAP_unstructed.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

#===================================================================================
# caption models - 1 show attend and tell !!!
#===================================================================================
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class Caption1(nn.Module):
    def __init__(self, Flags):
        super().__init__()
        self.encoder = smp.FPN(Flags.encoder, in_channels=3, encoder_weights='imagenet').encoder
        self.feat_dims = self.encoder.out_channels[-1]
        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.feat_dims, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, Flags.n_class)
        
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))

        # decoder
        self.encoder_dim = self.encoder.out_channels[-1]
        self.attention_dim = 512
        self.embed_dim = 512
        self.decoder_dim = 512
        self.vocab_size = Flags.word_map_len
        self.dropout = 0.5

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        outputs = dict()
        feat1 = self.encoder(x1)[-1]
        feat2 = self.encoder(x2)[-1]
        feat = feat1 + feat2

        feat_classify = self.avgpool_global(feat).view(feat.size(0), -1)
        feat_classify = self.dropout(feat_classify)
        feat_classify = self.fc1(feat_classify)
        feat_classify = self.relu_fc(feat_classify)
        y_classify = self.fc2(feat_classify)
        outputs.update({"classify_out": y_classify})

        feat_caption = self.avgpool(feat)
        encoder_out = feat_caption.permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) 

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()   
        
        # Create tensors to hold word predicion scores and alphas
        if torch.cuda.is_available():
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # return predictions, encoded_captions, decode_lengths, alphas, sort_ind
        outputs.update({"main_out": predictions})
        outputs.update({"encoded_captions": encoded_captions})
        outputs.update({"decode_lengths": decode_lengths})
        outputs.update({"alphas": alphas})
        outputs.update({"sort_ind": sort_ind})

        return outputs

    def sample(self, x1, x2, encoded_captions, caption_lengths, start_token):
        outputs = dict()
        feat1 = self.encoder(x1)[-1]
        feat2 = self.encoder(x2)[-1]
        feat = feat1 + feat2

        feat_classify = self.avgpool_global(feat).view(feat.size(0), -1)
        feat_classify = self.dropout(feat_classify)
        feat_classify = self.fc1(feat_classify)
        feat_classify = self.relu_fc(feat_classify)
        y_classify = self.fc2(feat_classify)
        outputs.update({"classify_out": y_classify})

        feat_caption = self.avgpool(feat)
        encoder_out = feat_caption.permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        start_token_ = torch.LongTensor([[start_token]] * encoder_out.shape[0]).cuda()
        embeddings = self.embedding(start_token_).squeeze()

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) 

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()   
        
        # Create tensors to hold word predicion scores and alphas
        if torch.cuda.is_available():
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            embeddings = self.embedding(preds.max(1)[1])

        outputs.update({"main_out": predictions})
        outputs.update({"encoded_captions": encoded_captions})
        outputs.update({"decode_lengths": decode_lengths})
        outputs.update({"alphas": alphas})
        outputs.update({"sort_ind": sort_ind})

        return outputs

    def sample_t(self, x1, x2, start_token):
        outputs = dict()
        feat1 = self.encoder(x1)[-1]
        feat2 = self.encoder(x2)[-1]
        feat = feat1 + feat2

        feat_classify = self.avgpool_global(feat).view(feat.size(0), -1)
        feat_classify = self.dropout(feat_classify)
        feat_classify = self.fc1(feat_classify)
        feat_classify = self.relu_fc(feat_classify)
        y_classify = self.fc2(feat_classify)
        outputs.update({"classify_out": y_classify})

        feat_caption = self.avgpool(feat)
        encoder_out = feat_caption.permute(0, 2, 3, 1)

        enc_image_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(enc_image_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        h, c = self.init_hidden_state(encoder_out)
        if torch.cuda.is_available():
            predictions = torch.zeros(enc_image_size, 82, vocab_size).cuda()
            alphas = torch.zeros(enc_image_size, 82, num_pixels).cuda()
        # Embedding
        start_token_ = torch.LongTensor([[start_token]] * encoder_out.shape[0]).cuda()
        embeddings = self.embedding(start_token_).squeeze()

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) 

        for t in range(82):
            awe, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c)) 
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
            embeddings = self.embedding(preds.max(1)[1])

        outputs.update({"main_out": predictions})
        return outputs

