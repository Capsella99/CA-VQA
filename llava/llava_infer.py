from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import torch
import time
import json
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def get_frames(filename):
    cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target = total_frames // 8
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("warning!!")
            break
        if i % target == 0:
            #rame = cv2.resize(frame, (224, 224))  # Resize frame to [224, 224]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = torch.tensor(frame)
            frames.append(frame)
            if len(frames) == 8:
                break
    cap.release()
    return torch.stack(frames)

def main(args):
    device = torch.device("cuda:0")
    path_prefix = args.data_dir
    file_path = "LLaVA/youtube_ugc_val.txt"
    label_path = "FAST-VQA-and-FasterVQA-dev/examplar_data_labels/YouTubeUGC/labels.txt"

    files = open(file_path,'r').read().split('\n')
    labels = open(label_path,'r').read().split('\n')
    label = {}
    for i in labels:
        e = i.split(",")
        label[e[0].strip()] = e[-1].strip()

    for i, f in enumerate(files):
        files[i] = str(path_prefix / f)

    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf").half().to(device)
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    datas = []
    
    for i in range(len(files)):
        video_frames = get_frames(files[i])
        prompt = f"USER: <video> The Video quality is ASSISTANT:"
        inputs = processor(text=prompt, videos=video_frames, return_tensors="pt").to(device)
        logits = model(**inputs).logits[:,-1]
        lgood, lpoor = logits[0,1781].item(), logits[0,6460].item()
        score = np.exp(lgood) / (np.exp(lgood) + np.exp(lpoor))
        label_key = files[i].split("/")[-1]
        datas.append({"file": files[i], "pred": score , "label": label[label_key]})
            
    save_to_json(datas, "llave_youtube_ugc_val.json")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/home/ethan/original_videos_h264",
    )
    parser.add_argument("--index", type=int, default=11)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)