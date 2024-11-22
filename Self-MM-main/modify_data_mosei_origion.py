import pickle

import numpy as np
from transformers import PROCESSOR_MAPPING

np.set_printoptions(threshold=np.inf)
data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSEI\Processed\unaligned_50.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['train'].keys())

# 创建新的回归标签
def create_classification_labels_new(subdata_name):
    classification_labels_new = []
    for label in data[subdata_name]['regression_labels']:
        if label < 0:
            classification_labels_new.append(0)
        elif label == 0:
            classification_labels_new.append(1)
        elif 0 < label :
            classification_labels_new.append(2)
    return np.array(classification_labels_new)  # 转换为 numpy 数组

classification_labels_new1=create_classification_labels_new('train')
classification_labels_new2=create_classification_labels_new('valid')
classification_labels_new3=create_classification_labels_new('test')
# 添加新的回归标签到数据中
data['train']['classification_labels_new'] = classification_labels_new1
data['valid']['classification_labels_new'] = classification_labels_new2
data['test']['classification_labels_new'] = classification_labels_new3

# 保存新的数据到新文件中
new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSEI\unaligned_new0.pkl'
with open(new_data_path, 'wb') as f:
    pickle.dump(data, f)

