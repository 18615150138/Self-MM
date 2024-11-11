import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils_copy.functions import dict_to_str
from utils_copy.metricsTop import MetricsTop
from trains_copy.multiTask.loss_function import get_loss

logger = logging.getLogger('MSA')

class SELF_MM():
    def __init__(self, args):
        assert args.train_mode == 'classification'
        self.args = args

    def do_train(self,model,dataloader):

        # 参数分组与优化器设置
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                              'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        annealing_step=self.args.annealing_step
        gamma = self.args.gamma
        device=self.args.device
        model.to(device)
        acc_max = 0

        print('acc_max',acc_max)

        for epoch in range(1, self.args.epochs + 1):
            print(f'====> {epoch}')
            for phase in ["train", "valid"]:
                if phase == "train":
                    print('now is batch_size train')
                    model.train()
                    with tqdm(dataloader['train']) as td:
                        for batch_data in td:
                            Y = batch_data['labels']['M'].to(self.args.device).long()
                            vision = batch_data['vision'].to(self.args.device)
                            audio = batch_data['audio'].to(self.args.device)
                            text = batch_data['text'].to(self.args.device)
                            #indexes = batch_data['index'].view(-1)
                            #cur_id = batch_data['id']
                            #ids.extend(cur_id)

                            if not self.args.need_data_aligned:
                                audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                                vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                            else:
                                audio_lengths, vision_lengths = 0, 0

                            evidences, evidence_a = model(text, (audio, audio_lengths), (vision, vision_lengths))
                            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes=3, annealing_step=annealing_step, gamma=gamma,device=device)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                elif phase == "valid":
                    print('now is batch_size valid')
                    model.eval()
                    num_correct, num_sample = 0, 0
                    with tqdm(dataloader['valid']) as td:
                        for batch_data in td:
                            Y = batch_data['labels']['M'].to(self.args.device).long()
                            vision = batch_data['vision'].to(self.args.device)
                            audio = batch_data['audio'].to(self.args.device)
                            text = batch_data['text'].to(self.args.device)
                            #indexes = batch_data['index'].view(-1)
                            #cur_id = batch_data['id']
                            #ids.extend(cur_id)

                            if not self.args.need_data_aligned:
                                audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                                vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                            else:
                                audio_lengths, vision_lengths = 0, 0

                            with torch.no_grad():
                                evidences, evidence_a = model(text, (audio, audio_lengths), (vision, vision_lengths))
                                _, Y_pre = torch.max(evidence_a, dim=1)
                                num_correct += (Y_pre == Y).sum().item()
                                num_sample += Y.shape[0]
                        print('num_correct:',num_correct,'num_sample',num_sample) #出问题了这里，晚上再debug
                        acc = num_correct / num_sample
                        if acc > acc_max:
                            acc_max = acc
                            acc_max_epoch = epoch
                            # save model
                            torch.save(model.cpu().state_dict(), self.args.model_save_path)
                            model.to(self.args.device)

            print('====> acc_max: {:.4f}'.format(acc_max))

        return acc_max

    def do_test(self, model, dataloader):
        print('now is all test')
        model.eval()
        num_correct, num_sample = 0, 0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    Y = batch_data['labels']['M'].to(self.args.device).long()
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0
                    evidences, evidence_a = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    _, Y_pre = torch.max(evidence_a, dim=1)
                    num_correct += (Y_pre == Y).sum().item()
                    num_sample += Y.shape[0]
                acc_test = num_correct / num_sample
        return acc_test