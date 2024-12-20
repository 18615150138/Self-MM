# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from six import text_type
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models_copy.subNets.BertTextEncoder import BertTextEncoder


__all__ = ['SELF_MM_copy']

#司师兄的网络
class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()

        #这块实际没运行
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))

        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.BatchNorm1d(num_classes)) #加了个正则化
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h

#RCML
class SELF_MM_copy(nn.Module):
    def __init__(self, args):

        #print("self_mm_copy")
        super(SELF_MM_copy, self).__init__()

        # 定义特征三个模态的特征提取网络，这块我不能改
        # text subnets
        self.aligned = args.need_data_aligned
        print('args.language',args.language)
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        # audio-vision subnets
        #args.feature_dims (768, 33, 709)
        audio_in, video_in = args.feature_dims[1:] #33, 709
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)


        # #下面这三个模态的post_text_dropout和post_text_layer_1是想把特征向量的维度尽量靠拢，防止维度大小差异过大
        # # the classify layer for text
        # self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        # self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        #
        # # the classify layer for audio
        # self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        # self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        #
        # # the classify layer for video
        # self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        # self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)



        # 定义3个证据网络，特征向量跑完这个网络可以得到每个模态的证据向量
        self.num_views=3
        self.num_classes = args.num_classes
        #print('self.num_classes', self.num_classes)
        dims = [[args.text_out],[args.audio_out],[args.video_out]]
        #dims = [[args.post_text_dim], [args.post_audio_dim], [args.post_video_dim]]
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        # 这个dims[i]是第i个视图的数据的维度，这里是定义了num_views个单模态网络


    def forward(self, text, audio, video):

        #print("self_mm_copy forward")
        audio, audio_lengths = audio
        #shape=(32,16)=(batch_size,audio_out)
        video, video_lengths = video
        #shape=(32,32)=(batch_size,video_out)
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]
        #shape=(32,768)=(batch_size,text_out)
        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)


        # # # text
        # text_h = self.post_text_dropout(text)  # (32,64)
        # text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # # audio
        # audio_h = self.post_audio_dropout(audio)  # (32,16)
        # audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # # vision
        # video_h = self.post_video_dropout(video)  # (32,32)
        # video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)
        #
        #
        # t=text_h
        # a=audio_h
        # v=video_h
        #
        # X=[t,a,v]
        X =[text,audio,video] #(32,768) (32,16) (32,32)
        evidences=dict()
        for v in range(self.num_views):
            evidences[v]=self.EvidenceCollectors[v](X[v])

        #ABF
        evidence_a=evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a)
        evidence_a=evidence_a/self.num_views

        # #WBF
        # u = dict()
        # weight_sum = 0
        # weight = dict()
        # for v in range(self.num_views):
        #     u[v] = self.num_classes / (self.num_classes + evidences[v].sum(dim=1))
        #     weight_sum += 1 - u[v]
        #
        # for v in range(self.num_views):
        #     weight[v] = (1 - u[v]) / weight_sum
        # evidence_a = evidences[0] * weight[0].unsqueeze(1)
        # for v in range(1, self.num_views):
        #     # print('evidences[v]*weight[v].unsqueeze(1)',evidences[v]*weight[v].unsqueeze(1))
        #     evidence_a += evidences[v] * weight[v].unsqueeze(1)

        return evidences,evidence_a




class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        #先确保lengths参数被正确地转移到CPU上，并且是int64类型
        lengths = lengths.cpu().to(torch.int64)

        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
