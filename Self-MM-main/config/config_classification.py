import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'self_mm': self.__SELF_MM
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = 'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    #'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/unaligned_new.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    #'train_samples': 16326,
                    'train_samples': 10620,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    #'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    #'num_classes': 3,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMSv2/unaligned_new.pkl'),
                    # 'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55),  # (text, audio, video)
                    'feature_dims': (768, 33, 709),  # (text, audio, video)
                    'train_samples': 2722,
                    # 'num_classes': 3,
                    'num_classes': 11,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    #'learning_rate_other': 1e-3,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 1e-3,
                    'weight_decay_audio': 1e-3,
                    'weight_decay_video': 1e-3,
                    'weight_decay_other': 1e-3,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.2,
                    'v_lstm_dropout': 0.2,
                    't_bert_dropout':0.2,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.2,
                    'post_text_dropout': 0.2,
                    'post_audio_dropout': 0.2,
                    'post_video_dropout': 0.2,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    #'batch_size': 32,
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    #'learning_rate_other': 1e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 1e-2,
                    'weight_decay_audio': 1e-2,
                    'weight_decay_video': 1e-2,
                    'weight_decay_other': 1e-2,
                    # feature subNets
                    #'a_lstm_hidden_size': 32,
                    #'v_lstm_hidden_size': 32,
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 16,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.2,
                    'v_lstm_dropout': 0.2,
                    't_bert_dropout':0.2,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.2,
                    'post_text_dropout': 0.2,
                    'post_audio_dropout': 0.2,
                    'post_video_dropout': 0.2,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    #'learning_rate_other': 1e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 1e-3,
                    'weight_decay_audio': 1e-3,
                    'weight_decay_video': 1e-3,
                    'weight_decay_other': 1e-3,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.2,
                    'v_lstm_dropout': 0.2,
                    't_bert_dropout':0.2,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.2,
                    'post_audio_dropout': 0.2,
                    'post_video_dropout': 0.2,
                    # res
                    'H': 1.0
                },
                'simsv2': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-4,
                    'learning_rate_audio': 1e-4,
                    'learning_rate_video': 1e-4,
                    # 'learning_rate_other': 1e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.01,
                    'weight_decay_audio': 0.1,
                    'weight_decay_video': 0.1,
                    'weight_decay_other': 0.1,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.2,
                    'v_lstm_dropout': 0.2,
                    't_bert_dropout': 0.2,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim': 64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.2,
                    'post_audio_dropout': 0.2,
                    'post_video_dropout': 0.2,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args