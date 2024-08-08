import pandas as pd
import matplotlib.pyplot as plt
import json
# 讀取資料
data = pd.read_csv('FAST-VQA-and-FasterVQA-dev/examplar_data_labels/LIVE_VQC/labels.txt', header=None, names=['file', 'col2', 'col3', 'mos'])

# 繪製MOS分佈圖表
plt.figure(figsize=(10, 6))
plt.hist(data['mos'], bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('MOS')
plt.ylabel('Frequency')
plt.title('Distribution of MOS')
plt.savefig('LIVE_VQC.png')


# 讀取資料
data = pd.read_csv('FAST-VQA-and-FasterVQA-dev/examplar_data_labels/KoNViD/labels.txt', header=None, names=['file', 'col2', 'col3', 'mos'])

# 繪製MOS分佈圖表
plt.figure(figsize=(10, 6))
plt.hist(data['mos'], bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('MOS')
plt.ylabel('Frequency')
plt.title('Distribution of MOS')
plt.savefig('KoNViD.png')


# 讀取資料
data = pd.read_csv('FAST-VQA-and-FasterVQA-dev/examplar_data_labels/YouTubeUGC/labels.txt', header=None, names=['file', 'col2', 'col3', 'mos'])

# 繪製MOS分佈圖表
plt.figure(figsize=(10, 6))
plt.hist(data['mos'], bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('MOS')
plt.ylabel('Frequency')
plt.title('Distribution of MOS')
plt.savefig('YouTubeUGC.png')

# 讀取JSON資料
with open('data/label_p.json', 'r') as f:
    data_dict = json.load(f)

# 將資料轉換為 DataFrame
data_list = [{'file': v['file'], 'mos': v['mos']} for v in data_dict.values()]
data = pd.DataFrame(data_list)

# 繪製MOS分佈圖表
plt.figure(figsize=(10, 6))
plt.hist(data['mos'], bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('MOS')
plt.ylabel('Frequency')
plt.title('Distribution of MOS')
plt.savefig('AVA.png')