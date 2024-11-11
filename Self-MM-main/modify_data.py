import pickle

import numpy as np
from transformers import PROCESSOR_MAPPING

np.set_printoptions(threshold=np.inf)
#data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMS\unaligned_39.pkl'
data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSI\Processed\unaligned_50.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['train'].keys())
print(data['train']['regression_labels'])
print(data['train']['classification_labels'])


def validate_labels(regression_labels, classification_labels):
    assert len(regression_labels) == len(classification_labels), "Lengths of labels do not match"

    for i in range(len(regression_labels)):
        reg_label = regression_labels[i]
        class_label = classification_labels[i]

        if reg_label < 0 and class_label != 0:
            print(f"Mismatch at index {i}: regression_label={reg_label}, classification_label={class_label}")
        elif reg_label == 0 and class_label != 1:
            print(f"Mismatch at index {i}: regression_label={reg_label}, classification_label={class_label}")
        elif reg_label > 0 and class_label != 2:
            print(f"Mismatch at index {i}: regression_label={reg_label}, classification_label={class_label}")



validate_labels(data['train']['regression_labels'], data['train']['classification_labels'])