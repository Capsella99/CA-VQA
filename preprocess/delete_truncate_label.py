import json

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

with open("data/label_p.json", 'r', encoding='utf-8') as file:
    datas = json.load(file)

"""
del datas['2234']
del datas['10632']
del datas['23848']
del datas['36266']
del datas['57942']
del datas['176140']
"""

save_to_json(datas, "data/label_p.json")