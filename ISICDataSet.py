import os

from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, preload=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.preload = preload

        if preload:
            self.image_cache = []
            for _, row in self.dataframe.iterrows():
                image_id = row['isic_id']
                image_path = os.path.join(image_dir, image_id + ".jpg")
                img = Image.open(image_path).convert('RGB')
                self.image_cache.append(img)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label = int(row['malignant'])
        if self.preload:
            image = self.image_cache[idx]
        else:
            image_id = row['isic_id']
            image_path = os.path.join(self.image_dir, image_id + ".jpg")
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
