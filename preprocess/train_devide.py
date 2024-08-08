import json
import random

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

# 讀取原始資料
with open("data/label_p.json", 'r', encoding='utf-8') as file:
    datas = json.load(file)

# 將鍵值打亂，保證隨機分配
keys = list(datas.keys())
random.shuffle(keys)

# 計算90%和10%的分界點
split_index = int(len(keys) * 0.95)

# 分成90%和10%的兩部分
train_keys = keys[:split_index]
test_keys = keys[split_index:]

# 分別保存兩部分資料
train_data = {key: datas[key] for key in train_keys}
test_data = {key: datas[key] for key in test_keys}

# 保存到不同的JSON文件中
save_to_json(train_data, "data/label_p_train.json")
save_to_json(test_data, "data/label_p_test.json")