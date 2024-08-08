import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# 計算 AVA MOS

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def main(args):
    data = {}
    with open(args.source_label_dir,'r') as f:
        for line in f:
            parts = line.strip().split()
            
            numbers = list(map(int, parts))
            
            index = numbers[0]
            file = numbers[1]
            
            # 計算MOS
            scores = np.array(numbers[2:12])
            total_votes = np.sum(scores)
            if total_votes > 0:
                mos = np.dot(scores, np.arange(1, 11)) / total_votes
                mos = round(mos, 2)
            else:
                mos = 0.0

            data[index] = {
                'file': file,
                'mos': mos
            }
        save_to_json(data, "data/label.json")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_label_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/data/ethan/AVA_dataset/AVA.txt",
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.label_dir.mkdir(parents=True,exist_ok=True)
    main(args)
