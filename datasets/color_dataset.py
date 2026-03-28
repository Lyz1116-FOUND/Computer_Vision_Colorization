from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.lab_utils import prepare_input_target


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ColorDataset(Dataset):
    """把彩色图变成 L 输入和 ab 标签。"""

    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_list = sorted(
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.image_list:
            raise ValueError(f"文件夹里没找到图片: {self.image_dir}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        rgb = np.array(image).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        l, ab = prepare_input_target(rgb)

        return {
            "L": l.squeeze(0),
            "ab": ab.squeeze(0),
            "path": str(image_path),
        }
