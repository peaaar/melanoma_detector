import os
from PIL import Image
from torch.utils.data import Dataset

PREFIX = "/Users/jinghanli/personalSpace/research/isic 2024"
ARCHIVE_PREFIX = "#"
IMAGE_DIR = f"{PREFIX}/ISIC_2024_Training_Input"
IMAGE_ARCHIVE_DIR = f"{PREFIX}/isic-archive-resized"

class ISICDataset(Dataset):
    def __init__(self, dataframe, transform=None, preload=False):
        """
        image_dirs: list of directories to search for images
        """
        self.dataframe = dataframe
        self.transform = transform
        self.preload = preload

        if preload:
            self.image_cache = []
            for _, row in self.dataframe.iterrows():
                image_id = str(row['isic_id'])
                image_dir = IMAGE_ARCHIVE_DIR if image_id.startswith(ARCHIVE_PREFIX) else IMAGE_DIR
                image_path = os.path.join(image_dir, image_id.replace(ARCHIVE_PREFIX, "") + ".jpg")
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
            image_path = self._find_image_path(image_id)
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
