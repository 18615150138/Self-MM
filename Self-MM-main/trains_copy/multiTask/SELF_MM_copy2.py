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
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
    def do_train(self, model, dataloader):
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


        epochs, best_epoch = 0, 0

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    Y = batch_data['labels']['M'].to(self.args.device).long().squeeze()
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0



                    evidences, evidence_a = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    _, Y_pre = torch.max(evidence_a, dim=1)

                    num_classes=self.args.num_classes
                    gamma = self.args.gamma
                    device = self.args.device
                    annealing_step = self.args.annealing_step

                    loss = get_log_loss(evidences, evidence_a, Y, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)
                    # backward
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    train_loss += loss.item()
                    y_pred.append(Y_pre.cpu())
                    y_true.append(Y.cpu())
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                                                                 epochs - best_epoch, epochs, self.args.cur_time,
                                                                 train_loss))


            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return  None

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    Y = batch_data['labels']['M'].to(self.args.device).long().squeeze()
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0


                    evidences, evidence_a = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    _, Y_pre = torch.max(evidence_a, dim=1)

                    num_classes = self.args.num_classes
                    gamma = self.args.gamma
                    device = self.args.device
                    annealing_step = self.args.annealing_step

                    loss = get_log_loss(evidences, evidence_a, Y, epoch, num_classes=num_classes,annealing_step=annealing_step, gamma=gamma, device=device)

                    eval_loss += loss.item()
                    y_pred.append(Y_pre.cpu())
                    y_true.append(Y.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(mode + "-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        return eval_results


