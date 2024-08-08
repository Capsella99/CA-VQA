import pandas as pd
from datasets import Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from pathlib import Path
from transformers import AdamW, get_scheduler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class CustomDataset(Dataset):
    def __init__(self, data, path_prefix):
        self.data = data
        self.path_prefix = path_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.path_prefix / (str(self.data.iloc[idx]['file']) + ".jpg")
        image = Image.open(img_path).resize((224, 224))
        text = self.data.iloc[idx]['prompt']
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

class CustomCLIPTrainer(Trainer):
    def __init__(self, model, args, train_dataset):
        super().__init__(model, args, train_dataset=train_dataset)

        self.optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        num_update_steps_per_epoch = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(len(logits_per_image)).to(device)
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        return (loss, outputs) if return_outputs else loss


def evaluate_model(model, dataset, batch_size=192):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
    model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            preds_img = logits_per_image.argmax(dim=1)
            preds_txt = logits_per_text.argmax(dim=1)

            labels = torch.arange(len(preds_img)).to(device)

            num_correct += (preds_img == labels).sum().item()
            num_correct += (preds_txt == labels).sum().item()
            num_samples += len(preds_img) + len(preds_txt)

    accuracy = num_correct / num_samples
    return accuracy

def preprocess_function(data):
    img_paths = [path_prefix / (str(file_path)+".jpg") for file_path in data['file']]
    images = [Image.open(img_path).resize((224, 224)) for img_path in img_paths]
    inputs = processor(text=data['prompt'], images=images, return_tensors="pt", padding=True, max_length=77)
    return inputs

def main(args):
    train_df = pd.read_json('data/label_p_train.json').transpose().drop(columns=['rating'])
    train_dataset = CustomDataset(train_df, path_prefix)

    val_df = pd.read_json('data/label_p_test.json').transpose().drop(columns=['rating'])
    val_dataset = CustomDataset(val_df, path_prefix)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=192,
        num_train_epochs=6,
        save_steps=253,
        save_total_limit=10,
        learning_rate=5e-6,
        weight_decay=0.2,
        gradient_accumulation_steps=5,
        logging_steps=253,
    )

    trainer = CustomCLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    val_accuracy = evaluate_model(model, val_dataset)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

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
    global path_prefix
    my_args = parse_args()
    path_prefix = my_args.data_dir
    main(my_args)
