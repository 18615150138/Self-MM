import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)
new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSI\unaligned_new.pkl'

with open(new_data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['train_new'].keys())
print(data['valid_new'].keys())
print(data['test_new'].keys())

# 提取分类标签
train_labels = data['train_new']['classification_labels']
valid_labels = data['valid_new']['classification_labels']
test_labels = data['test_new']['classification_labels']

# 将 numpy 数组转换为 pandas Series
train_labels_series = pd.Series(train_labels)
valid_labels_series = pd.Series(valid_labels)
test_labels_series = pd.Series(test_labels)

# 计算每个数据集每个类别的样本数量
train_counts = train_labels_series.value_counts()
valid_counts = valid_labels_series.value_counts()
test_counts = test_labels_series.value_counts()
print(train_counts)
print(valid_counts)
print(test_counts)

# 创建单独的DataFrame
train_df = pd.DataFrame({'Set': ['train'] * len(train_labels), 'Classification Labels': train_labels})
valid_df = pd.DataFrame({'Set': ['valid'] * len(valid_labels), 'Classification Labels': valid_labels})
test_df = pd.DataFrame({'Set': ['test'] * len(test_labels), 'Classification Labels': test_labels})


def plot_trends(df, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Classification Labels', hue='Set', data=df)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Classification Labels', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Set')
    plt.show()

# 绘制所有数据集的分类标签分布图，并按数据集区分颜色
combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
plot_trends(combined_df, 'Overall Distribution of Classification Labels Across All Sets')

