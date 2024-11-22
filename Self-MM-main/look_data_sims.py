import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMS\unaligned_39_new.pkl'

with open(new_data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['train'].keys())
print(data['valid'].keys())
print(data['test'].keys())
print(data['all'].keys())

# 提取分类标签
train_labels = data['train']['classification_labels']
valid_labels = data['valid']['classification_labels']
test_labels = data['test']['classification_labels']
all_labels = data['all']['classification_labels']


# 将 train_labels 转换为 pandas Series
train_labels_series = pd.Series(train_labels)

# 使用 value_counts() 统计每个类别的样本个数，并按类别号排序
label_counts = train_labels_series.value_counts().sort_index()
print("Sample counts for each classification label in the training set:")
print(label_counts)


# 创建单独的DataFrame
train_df = pd.DataFrame({'Set': ['train'] * len(train_labels), 'Classification Labels': train_labels})
valid_df = pd.DataFrame({'Set': ['valid'] * len(valid_labels), 'Classification Labels': valid_labels})
test_df = pd.DataFrame({'Set': ['test'] * len(test_labels), 'Classification Labels': test_labels})
all_df = pd.DataFrame({'Set': ['all'] * len(all_labels), 'Classification Labels':all_labels})


def plot_trends(df, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Classification Labels', hue='Set', data=df)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Classification Labels', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Set')
    plt.show()

# 绘制所有数据集的分类标签分布图，并按数据集区分颜色
combined_df = pd.concat([train_df, valid_df, test_df,all_df], ignore_index=True)
plot_trends(combined_df, 'Overall Distribution of Classification Labels Across All Sets')

