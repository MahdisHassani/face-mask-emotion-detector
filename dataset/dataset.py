import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiTaskDataset(Dataset):
    def __init__(self, mask_dir, emotion_img_dir, emotion_csv, transform=None):
        self.transform = transform
        self.data = []

        # -------- MASK DATA --------
        with_mask_dir = os.path.join(mask_dir, "with_mask")
        without_mask_dir = os.path.join(mask_dir, "without_mask")

        for file in os.listdir(with_mask_dir):
            if file.endswith(".jpg"):
                img_path = os.path.join(with_mask_dir, file)
                self.data.append((img_path, 1, 0))

        for file in os.listdir(without_mask_dir):
            if file.endswith(".jpg"):
                img_path = os.path.join(without_mask_dir, file)
                self.data.append((img_path, 0, 0))

        # -------- EMOTION DATA --------
        df = pd.read_csv(emotion_csv)

        emotion_map = {
            "happiness": 0,
            "neutral": 1
        }

        for _, row in df.iterrows():
            emotion = str(row['emotion']).lower()
            if emotion not in emotion_map:
                continue

            img_path = os.path.join(emotion_img_dir, row['image'])
            if not os.path.exists(img_path):
                continue

            label = emotion_map[emotion]
            self.data.append((img_path, label, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, task_type = self.data[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, task_type