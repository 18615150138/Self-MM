import pickle
from collections import Counter
import numpy as np
import pandas as pd
from transformers import PROCESSOR_MAPPING
from sklearn.utils import resample
from collections import defaultdict

np.set_printoptions(threshold=np.inf)
data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMS\unaligned_39.pkl'
#data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSI\Processed\unaligned_50.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)


print(data.keys())
print(data['train'].keys())
print(data['valid'].keys())
print(data['test'].keys())
# print(len(data['train']['raw_text']))
# print(len(data['valid']['raw_text']))
# print(len(data['test']['raw_text']))
# print(data['train']['raw_text'][0])
# print(data['valid']['raw_text'][0])
# print(data['test']['raw_text'][0])
# 初始化一个空字典来存储所有数据
all_data = defaultdict(list)
# 定义要合并的数据集
datasets = ['train', 'valid', 'test']
# 遍历每个数据集并将数据合并到 all_data 中
for dataset in datasets:
    for key in data[dataset].keys():
        all_data[key].extend(data[dataset][key])
# 将合并后的数据存储在 data['all'] 中
data['all'] = dict(all_data)
# 打印结果以确认
print(data.keys())
print(data['all'].keys())
all_labels = data['all']['classification_labels']
# 将 train_labels 转换为 pandas Series
all_labels_series = pd.Series(all_labels)
# 使用 value_counts() 统计每个类别的样本个数，并按类别号排序
label_counts = all_labels_series.value_counts().sort_index()
print("Sample counts for each classification label in the training set:")
print(label_counts)




# # 获取所有分类标签
# labels_V = data['all']['classification_labels_V']
# labels_A = data['all']['classification_labels_A']
# labels_T = data['all']['classification_labels_T']
#
# # 初始化一个空列表来存储不一致的样本下标
# labeldiff_id1 = [] #最严重的冲突
# labeldiff_id2 = [] #次严重的冲突
# labelsame_id=[] #不冲突
# # 遍历所有样本的下标
# for i in range(len(labels_V)):
#     diff_degree=0
#     if labels_T[i]!=labels_A[i]:
#         diff_degree+=1
#     if labels_T[i]!=labels_V[i]:
#         diff_degree+=1
#     if labels_A[i]!=labels_V[i]:
#         diff_degree+=1
#
#     if diff_degree==3:
#         labeldiff_id1.append(i)
#     elif diff_degree==2:
#         labeldiff_id2.append(i)
#     else:
#         labelsame_id.append(i)
#
# # 打印不一致的样本下标
# # print(len(labelsame_id)," 0chongtu samples:", labelsame_id)
# # print(len(labeldiff_id1)," 3chongtu samples:", labeldiff_id1)
# # print(len(labeldiff_id2)," 2chongtu samples:", labeldiff_id2)
# print(len(labelsame_id))
# print(len(labeldiff_id1))
# print(len(labeldiff_id2))






# # 保存新的数据到新文件中
# new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMS\unaligned_39_new.pkl'
# with open(new_data_path, 'wb') as f:
#     pickle.dump(data, f)


