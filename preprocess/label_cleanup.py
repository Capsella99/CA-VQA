import json
import os.path
from argparse import ArgumentParser
from pathlib import Path

# 從label中刪除不存在的檔案

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


def main(args):
    path_prefix = args.data_dir
    with open("data/label_r.json", 'r', encoding='utf-8') as file:
        datas = json.load(file)

    keys = []
    for key in datas:
        data = datas[key]
        path = path_prefix / (str(data['file']) + ".jpg")
        if os.path.isfile(path) is not True:
            print(f"{data['file']} is not exist")
            keys.append(key)        

    for key in keys:  
        del datas[key]

    save_to_json(datas, "data/label_r.json")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/data/ethan/AVA_dataset/image/",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
