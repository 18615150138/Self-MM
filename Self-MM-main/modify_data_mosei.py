import gc
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

np.set_printoptions(threshold=np.inf)
new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSEI\Processed\unaligned_50.pkl'
save_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSEI\unaligned_new.pkl'
with open(new_data_path, 'rb') as f:
    data = pickle.load(f)

# 提取分类标签
train_labels = data['train']['classification_labels']
valid_labels = data['valid']['classification_labels']
test_labels = data['test']['classification_labels']

# 将 numpy 数组转换为 pandas Series
train_labels_series = pd.Series(train_labels)
valid_labels_series = pd.Series(valid_labels)
test_labels_series = pd.Series(test_labels)

# 计算每个数据集每个类别的样本数量
train_counts = train_labels_series.value_counts()
valid_counts = valid_labels_series.value_counts()
test_counts = test_labels_series.value_counts()

# 使用欠采样技术生成新的训练集
min_class_count_train = min(train_counts)
min_class_count_valid = min(valid_counts)
min_class_count_test = min(test_counts)
# 获取每个类别的索引
train_indices = {label: np.where(train_labels == label)[0] for label in train_counts.index}
valid_indices = {label: np.where(valid_labels == label)[0] for label in valid_counts.index}
test_indices = {label: np.where(test_labels == label)[0] for label in test_counts.index}
# 随机选择每个类别的min_class_count个样本
undersampled_indices_train = np.concatenate([np.random.choice(indices, min_class_count_train, replace=False) for indices in train_indices.values()])
undersampled_indices_valid = np.concatenate([np.random.choice(indices, min_class_count_valid, replace=False) for indices in valid_indices.values()])
undersampled_indices_test = np.concatenate([np.random.choice(indices, min_class_count_test, replace=False) for indices in test_indices.values()])
# 创建新的训练集
data['train_new'] = {key: value[undersampled_indices_train] if isinstance(value, np.ndarray) else value for key, value in data['train'].items()}
data['valid_new'] = {key: value[undersampled_indices_valid] if isinstance(value, np.ndarray) else value for key, value in data['valid'].items()}
data['test_new'] = {key: value[undersampled_indices_test] if isinstance(value, np.ndarray) else value for key, value in data['test'].items()}

# 创建新的字典 data1，只包含新生成的训练、验证和测试集
data1 = {
    'train': data['train_new'],
    'valid': data['valid_new'],
    'test': data['test_new']
}

del data
gc.collect()


with open(save_data_path, 'wb') as f:
    pickle.dump(data1, f)

# #报错，内存不足，因此转用分批次写入
# # 假设每个批次的大小为 batch_size
# batch_size = 16
#
# def save_batch(data, file_path):
#     with open(file_path, 'ab') as f:  # 使用 'ab' 模式以追加方式写入文件
#         pickle.dump(data, f)
#     print(f"Batch saved to {file_path}")
#
# # 获取数据集大小
# train_size = len(data1['train_new']['classification_labels'])
# valid_size = len(data1['valid_new']['classification_labels'])
# test_size = len(data1['test_new']['classification_labels'])
#
# # 计算需要多少个批次
# num_batches_train = (train_size + batch_size - 1) // batch_size
# num_batches_valid = (valid_size + batch_size - 1) // batch_size
# num_batches_test = (test_size + batch_size - 1) // batch_size
#
# # 保存训练集
# for i in range(num_batches_train):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, train_size)
#     batch_data = {key: value[start_idx:end_idx] if isinstance(value, np.ndarray) else value for key, value in data1['train_new'].items()}
#     save_batch(batch_data, save_data_path)
#
# # 保存验证集
# for i in range(num_batches_valid):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, valid_size)
#     batch_data = {key: value[start_idx:end_idx] if isinstance(value, np.ndarray) else value for key, value in data1['valid_new'].items()}
#     save_batch(batch_data, save_data_path)
#
# # 保存测试集
# for i in range(num_batches_test):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, test_size)
#     batch_data = {key: value[start_idx:end_idx] if isinstance(value, np.ndarray) else value for key, value in data1['test_new'].items()}
#     save_batch(batch_data, save_data_path)

