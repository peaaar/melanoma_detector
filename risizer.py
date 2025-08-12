from PIL import Image
import os

from ISICDataset import PREFIX, IMAGE_ARCHIVE_DIR

DEST_DIR = f"{PREFIX}/isic-archive-resized"

def resize():
    os.makedirs(DEST_DIR, exist_ok=True)

    for fname in os.listdir(IMAGE_ARCHIVE_DIR):
        if fname.endswith(".jpg"):
            img = Image.open(os.path.join(IMAGE_ARCHIVE_DIR, fname)).convert("RGB")
            img = img.resize((224, 224))
            img.save(os.path.join(DEST_DIR, fname))

if __name__ == "__main__":
    resize()