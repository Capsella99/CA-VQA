import time
import json

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

with open("data/label_p.json", 'r', encoding='utf-8') as file:
    datas = json.load(file)

for key in datas:
    data = datas[key]
    data['prompt'] = data["prompt"].split("ASSISTANT:")[-1].strip()

save_to_json(datas, "data/label_p.json")