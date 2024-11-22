import pickle
from collections import Counter
import numpy as np
from transformers import PROCESSOR_MAPPING
from sklearn.utils import resample


np.set_printoptions(threshold=np.inf)
data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMSv2\unaligned.pkl'
#data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\MOSI\Processed\unaligned_50.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['train'].keys())
print(data['valid'].keys())
print(data['test'].keys())





# # 保存新的数据到新文件中
# new_data_path = r'E:\yanyi\code of paper\Self-MM-main\Self-MM-main\Dataset\SIMSv2\unaligned_new.pkl'
# with open(new_data_path, 'wb') as f:
#     pickle.dump(data, f)


