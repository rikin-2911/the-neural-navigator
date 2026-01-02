# data_loader for neural navigator
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Vocabulary
VOCAB = {
    "go": 0, "to": 1, "the": 2,
    "red": 3, "green": 4, "blue": 5,
    "circle": 6, "square": 7, "triangle": 8
}
PAD_IDX = 9
MAX_LEN = 5

class NavigationDataset(Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        self.files = sorted([
            f for f in os.listdir(self.img_dir) if f.endswith(".png")
        ])

        self.transform = T.ToTensor()

    def tokenize(self, text):
        """
        Converts text instruction to fixed-length token IDs
        """
        tokens = text.lower().replace(",", "").split()
        ids = [VOCAB.get(tok, PAD_IDX) for tok in tokens]
        ids = ids[:MAX_LEN] + [PAD_IDX] * (MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # (3, 128, 128)

        
        ann_path = os.path.join(
            self.ann_dir,
            img_name.replace(".png", ".json")
        )
        ann = json.load(open(ann_path))

       
        text_ids = self.tokenize(ann["text"])

        
        path = torch.tensor(
            ann["path"],
            dtype=torch.float32
        ) / 128.0  

        return img, text_ids, path

    def __len__(self):
        return len(self.files)