import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class PedsXrayRSNA2017(Dataset):
    def __init__(self, annotations_file, img_dir, exclude_ids=None, transform=None, target_transform=None):
        self.img_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # exclude unwanted ids
        if exclude_ids is not None:
            original_len = len(self.img_df)
            self.img_df = self.img_df[~(self.img_df["id"].isin(exclude_ids))]
            assert self.__len__() == original_len - len(exclude_ids)

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        image_id = str(self.img_df["id"].iloc[idx])
        img_path = os.path.join(self.img_dir, image_id + ".png")
        image = Image.open(img_path).convert('RGB')
        label = self.img_df["boneage"].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.cuda()
        return image, label