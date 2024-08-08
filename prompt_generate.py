from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
import torch
import time
import json
import os
import logging
from argparse import ArgumentParser
from pathlib import Path

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w')
    logger = logging.getLogger()

    device_ = "cuda:0"
    device = torch.device("cuda:0")
    path_prefix = args.data_dir

    label_path = "data/label_p.json"
    if not os.path.isfile(label_path):
        label_path = "data/label_r.json"

    with open(label_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                        load_in_4bit=True, 
                                                        device_map=device_,  
                                                        bnb_4bit_compute_dtype=torch.float16)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    index = args.index
    keys = list(datas.keys())
    key_index = keys.index(str(index))
    keys = keys[key_index:(index+220)]

    start = time.time()
    processed=0
    for key in keys:
        data = datas[key]
        rated = data['rating']
        image = Image.open(path_prefix / (str(data['file']) + ".jpg"))

        prompt = f"USER: <image>\nWhat you get is an image of {rated} aesthetics. Evaluate image aesthetics based on factors such as content, color, lighting, and composition. ASSISTANT:"

        try:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            generate_ids = model.generate(**inputs, max_new_tokens=170)
            reply = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if data.get("prompt", -1) == -1:
                data['prompt'] = reply
            else:
                logger.info(f"嘗試覆寫prompt:{key}")
            processed += 1
        except:
            print(f"{key} is truncated file, please delete it!")
            logger.info(f"{key} is truncated file, please delete it!")
            continue
        
        if processed % 200 == 0:
            end = time.time()
            print(f"已處理{processed}張圖片, 本輪耗時{end-start}")
            logger.info(f"已處理{processed}張圖片, 本輪耗時{end-start}")
            start = time.time()
            save_to_json(datas, "data/label_p.json")
            
    save_to_json(datas, "data/label_p.json")
    logger.info(f"End!")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/data/ethan/AVA_dataset/image/",
    )
    parser.add_argument("--index", type=int, default=11)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
