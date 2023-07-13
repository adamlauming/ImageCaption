'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-21 19:55:48
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

class SentenceLSTM(nn.Module):
    def __init__(self, embed_size=2048, hidden_size=512, attention_dim=512, num_layers=2, dropout=0.3, momentum=0.1):
        super(SentenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.encoder_att = nn.Linear(embed_size, attention_dim) 
        self.decoder_att = nn.Linear(hidden_size, attention_dim)  
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 

        self.W_t_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size, out_features=hidden_size, bias=True)   
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size, out_features=2, bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.init_wordh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.init_wordc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

        self.init_wordh.weight.data.uniform_(-0.1, 0.1)
        self.init_wordh.bias.data.fill_(0)
        self.init_wordc.weight.data.uniform_(-0.1, 0.1)
        self.init_wordc.bias.data.fill_(0)

    def forward(self, visual_feature, pre_hidden, last_state):
        att1 = self.encoder_att(visual_feature)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(pre_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        ctx = (visual_feature * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, last_state)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(self.W_stop_s(hidden_state)))

        h0_word = self.tanh(self.init_wordh(self.dropout(topic))).transpose(0, 1)
        c0_word = self.tanh(self.init_wordc(self.dropout(topic))).transpose(0, 1)
        return topic, p_stop, hidden_state, states, h0_word, c0_word


class WordLSTM(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, vocab_size=512, num_layers=1, n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.__init_weights()
        self.n_max = n_max

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic, captions, states):
        """
        state is initialized from topic_vec
        """
        embeddings = self.embed(captions)  # (bs, embed_size)
        embeddings = torch.cat((topic, embeddings.unsqueeze(1)), 1)  #
        output, states = self.lstm(embeddings, states)
        outputs = self.linear(output[:, -1, :])
        return outputs, states

    def sample(self, topic, start_tokens, states):
        sampled_ids = np.zeros((np.shape(states[0])[1], self.n_max))
        sampled_ids = torch.Tensor(sampled_ids).long().cuda()
        sampled_ids[:, 0] = start_tokens.view(-1, )
        predicted = start_tokens.squeeze(1)
        for i in range(1, self.n_max):
            outputs, state_t = self.forward(topic, predicted, states)
            states = state_t
            predicted = torch.max(outputs, 1)[1]  # argmax predicted.shape=(batch_size, 1)
            sampled_ids[:, i] = predicted
            if predicted.cpu().numpy() == word_map['<end>']:
                break

        return sampled_ids
    

#===================================================================================
# caption models - 4 HRNN-SCM!
#===================================================================================
class Caption4(nn.Module):
    def __init__(self, Flags):
        super().__init__()
        self.encoder = smp.FPN(Flags.encoder, in_channels=3, encoder_weights='imagenet').encoder
        self.feat_dims = self.encoder.out_channels[-1]
        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.feat_dims, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, Flags.n_class)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout(0.5)

        # sentence LSTM
        self.encoder_dim = self.encoder.out_channels[-1]
        self.hidden_dim = 512
        self.sentence_LSTM = SentenceLSTM()
        self.vocab_size = Flags.word_map_len
        self.word_LSTM = WordLSTM(vocab_size=self.vocab_size)

        self.init_h = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.hidden_dim)
        
        self.sementic_features_dim = 512
        self.sementic_embed = nn.Embedding(Flags.n_class, self.sementic_features_dim)

        self.scm = SCM(in_channels=self.feat_dims, inter_channels=512)
        self.fc_scm = nn.Linear(512+self.feat_dims, Flags.n_class)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_nor = nn.Conv2d(512+self.feat_dims, self.feat_dims, kernel_size=1)


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        h_init = torch.zeros(2, h.shape[0], h.shape[1]).cuda()
        c_init = torch.zeros(2, c.shape[0], c.shape[1]).cuda()
        h_init[0,:,:], c_init[0,:,:] = h, c
        h_init[1,:,:], c_init[1,:,:] = h, c 
        return h_init, c_init

    def forward(self, x1, x2, context):
        outputs = dict()
        feat1 = self.encoder(x1)[-1]
        feat2 = self.encoder(x2)[-1]

        xx1, xx2 = self.scm(feat1, feat2)
        feat = self.conv_nor(xx1 + xx2)
        xx1 = self.avgpool_1(xx1)
        xx2 = self.avgpool_2(xx2)
        x = xx1 + xx2
        x = x.view(x.size(0), -1)
        y_classify = self.fc_scm(x)

        # feat = feat1 + feat2
        # feat_classify = self.avgpool_global(feat).view(feat.size(0), -1)
        # feat_classify = self.dropout(feat_classify)
        # feat_classify = self.fc1(feat_classify)
        # feat_classify = self.relu_fc(feat_classify)
        # y_classify = self.fc2(feat_classify)
        outputs.update({"classify_out": y_classify})

        # tags = torch.sigmoid(y_classify)
        # semantic_features = self.sementic_embed(torch.topk(tags, 10)[1])

        feat_caption = self.avgpool(feat)
        encoder_out = feat_caption.view(feat_caption.size(0), feat_caption.size(1), -1).transpose(1, 2)  # (batch_size, 49, 2048)
        # avg_encoder = self.avgpool_global(feat).view(feat.size(0), -1)
        
        prev_hidden_states = torch.zeros(x1.shape[0], 1, self.hidden_dim).cuda()
        sentence_states = None
        
        p_stop_list = []
        word_list = []
        for sentence_index in range(context.shape[1]):
            topic, p_stop, hidden_states, states, h0, c0 = self.sentence_LSTM(encoder_out, prev_hidden_states, sentence_states)
            prev_hidden_states = hidden_states
            sentence_states = states
            last_hidden = (h0, c0)
            p_stop_list.append(p_stop.squeeze())

            s_word_list = []
            for word_index in range(context.shape[2] - 1): 
                words, hidden = self.word_LSTM.forward(topic, context[:, sentence_index, word_index], last_hidden)
                last_hidden = hidden
                s_word_list.append(words)
            word_list.append(s_word_list)

        outputs.update({"stop_out": p_stop_list})
        outputs.update({"word_out": word_list})
        return outputs

    def sample(self, x1, x2, start_token):
        outputs = dict()
        feat1 = self.encoder(x1)[-1]
        feat2 = self.encoder(x2)[-1]

        xx1, xx2 = self.scm(feat1, feat2)
        feat = self.conv_nor(xx1 + xx2)
        xx1 = self.avgpool_1(xx1)
        xx2 = self.avgpool_2(xx2)
        x = xx1 + xx2
        x = x.view(x.size(0), -1)
        y_classify = self.fc_scm(x)

        # feat = feat1 + feat2
        # feat_classify = self.avgpool_global(feat).view(feat.size(0), -1)
        # feat_classify = self.dropout(feat_classify)
        # feat_classify = self.fc1(feat_classify)
        # feat_classify = self.relu_fc(feat_classify)
        # y_classify = self.fc2(feat_classify)
        outputs.update({"classify_out": y_classify})

        # tags = torch.sigmoid(y_classify)
        # semantic_features = self.sementic_embed(torch.topk(tags, 10)[1])

        feat_caption = self.avgpool(feat)
        encoder_out = feat_caption.view(feat_caption.size(0), feat_caption.size(1), -1).transpose(1, 2)  # (batch_size, 49, 2048)
        # avg_encoder = self.avgpool_global(feat).view(feat.size(0), -1)

        prev_hidden_states = torch.zeros(x1.shape[0], 1, self.hidden_dim).cuda()
        sentence_states = None

        p_stop_list = []
        word_list = []
        for sentence_index in range(10):
            topic, p_stop, hidden_states, states, h0, c0 = self.sentence_LSTM(encoder_out, prev_hidden_states, sentence_states)
            prev_hidden_states = hidden_states
            sentence_states = states
            p_stop_list.append(p_stop.squeeze(1))
            p_stop = torch.max(p_stop.squeeze(1), 1)[1]

            start_tokens = np.zeros((topic.shape[0], 1))
            start_tokens[:, 0] = start_token
            start_tokens = torch.LongTensor(start_tokens).cuda()
            sampled_ids = self.word_LSTM.sample(topic, start_tokens, (h0, c0))
            sampled_ids = sampled_ids * p_stop
            word_list.append(sampled_ids.squeeze())
        outputs.update({"stop_out": p_stop_list})
        outputs.update({"word_out": word_list})
        return outputs

