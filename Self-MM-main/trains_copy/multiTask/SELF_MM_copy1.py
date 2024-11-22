import csv
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
from trains_copy.multiTask.loss_function import get_loss,get_loss1,get_log_loss

logger = logging.getLogger('MSA')

class SELF_MM():
    def __init__(self, args):
        assert args.train_mode == 'classification'
        self.args = args

    def do_train(self,model,dataloader):

        # 参数分组与优化器设置
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        model1=model.Model
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
        #为不同的参数组设置了不同的学习率和权重衰减（weight decay），这实际上已经包含了L2正则化的效果。Weight Decay是在优化器(Optimizer)上,可以推到他的效果和在Loss上加L2正则化一样。
        #weight_decay通常取1e-3，如果要尝试的话，一般也就是1e-2, 1e-3, 1e-4 这些选项。
        #权重衰退通常不对bias做。但通常bias做不做权重衰退其实效果差不多，不过最好不要做。
        #weight_decay取值越大，对抑制模型的强度越大。但这并不说明越大越好，太大的话，可能会导致模型欠拟合。

        optimizer = optim.Adam(optimizer_grouped_parameters)

        annealing_step=self.args.annealing_step
        gamma = self.args.gamma
        device=self.args.device
        num_classes=self.args.num_classes
        num_views=3
        model.to(device)
        acc_max = 0

        print('acc_max',acc_max)

        # 打开CSV文件进行写入
        with open('E:\\yanyi\\code of paper\\Self-MM-main\\Self-MM-main\\results_copy\\result_sims_edllogloss_abf.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['Epoch', 'Phase', 'Num Correct', 'Num Sample', 'Accuracy','Loss'])

            for epoch in range(1, self.args.epochs + 1):
                print(f'====> {epoch}')
                for phase in ["train", "valid"]:
                    if phase == "train":
                        print('now is batch_size train')
                        model.train()
                        num_correct, num_sample = 0, 0
                        class_counts1 = {0: 0, 1: 0, 2: 0}
                        class_counts2 = {0: 0, 1: 0, 2: 0}
                        train_loss=0
                        with tqdm(dataloader['train']) as td:
                            for batch_data in td:
                                train_batch_data = batch_data
                                Y = batch_data['labels']['M'].to(self.args.device).long().squeeze()
                                if self.args.datasetName == 'sims':
                                    Y_T=batch_data['labels']['T'].to(self.args.device).long().squeeze()
                                    Y_A = batch_data['labels']['T'].to(self.args.device).long().squeeze()
                                    Y_V = batch_data['labels']['T'].to(self.args.device).long().squeeze()

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
                                #Y_pre1=Y_pre
                                for y in Y:
                                    class_counts1[y.item()] += 1
                                for y_pre in Y_pre:
                                    class_counts2[y_pre.item()] += 1
                                num_correct += (Y_pre == Y).sum().item()
                                num_sample += Y.shape[0]

                                loss = get_log_loss(evidences, evidence_a, Y, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)
                                # if self.args.datasetName == 'sims':
                                #     loss = get_loss1(evidences, evidence_a, Y,Y_T,Y_A,Y_V, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)
                                # else:
                                #     loss = get_loss(evidences, evidence_a, Y, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)


                                train_loss+=loss
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                            train_acc = num_correct / num_sample
                            train_loss/=len(dataloader['train'])
                            print('train: num_correct: {}, num_sample: {}, train_acc: {:.4f}, train_loss: {:.4f}'.format(num_correct,
                                                                                                     num_sample,
                                                                                                     train_acc,train_loss))
                            print('Class counts - Y:', class_counts1)
                            print('Class counts - Y_pre:', class_counts2)
                            writer.writerow([epoch, 'train', num_correct, num_sample, train_acc,train_loss])

                    elif phase == "valid":
                        print('now is batch_size valid')
                        model.eval()
                        num_correct, num_sample = 0, 0
                        class_counts1 = {0: 0, 1: 0, 2: 0}
                        class_counts2 = {0: 0, 1: 0, 2: 0}
                        valid_loss=0
                        with torch.no_grad():
                            with tqdm(dataloader['valid']) as td:
                                for batch_data in td:
                                    valid_batch_data = batch_data
                                    Y = batch_data['labels']['M'].to(self.args.device).long().squeeze()
                                    vision = batch_data['vision'].to(self.args.device)
                                    audio = batch_data['audio'].to(self.args.device)
                                    text = batch_data['text'].to(self.args.device)

                                    if not self.args.need_data_aligned:
                                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                                    else:
                                        audio_lengths, vision_lengths = 0, 0

                                    evidences, evidence_a = model(text, (audio, audio_lengths),
                                                                  (vision, vision_lengths))
                                    _, Y_pre = torch.max(evidence_a, dim=1)

                                    for y in Y:
                                        class_counts1[y.item()] += 1
                                    for y_pre in Y_pre:
                                        class_counts2[y_pre.item()] += 1
                                    num_correct += (Y_pre == Y).sum().item()
                                    num_sample += Y.shape[0]

                                    loss = get_log_loss(evidences, evidence_a, Y, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)

                                    valid_loss+=loss

                        valid_acc = num_correct / num_sample
                        valid_loss/=len(dataloader['valid'])
                        print('valid: num_correct: {}, num_sample: {}, valid_acc: {:.4f}, valid_loss: {:.4f}'.format(num_correct, num_sample,valid_acc,loss))
                        print('Class counts - Y:', class_counts1)
                        print('Class counts - Y_pre:', class_counts2)
                        writer.writerow([epoch, 'valid', num_correct, num_sample, valid_acc,valid_loss])
                        if valid_acc > acc_max:
                            acc_max = valid_acc
                            acc_max_epoch = epoch
                            # save model
                            torch.save(model.cpu().state_dict(), self.args.model_save_path)
                            model.to(self.args.device)

                print('====> acc_max: {:.4f}'.format(acc_max))
                writer.writerow([epoch, 'overall', '', '', acc_max])

        return acc_max

    def do_test(self, model, dataloader):
        print('now is test')
        model.eval()
        num_correct, num_sample = 0, 0
        with torch.no_grad(): #这个放外面更好
            with tqdm(dataloader['test']) as td:
                for batch_data in td:

                    Y = batch_data['labels']['M'].to(self.args.device).long().squeeze()
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


