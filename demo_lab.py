import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.lab_utils import (
    denormalize_lab,
    lab_to_rgb,
    merge_l_ab,
    prepare_input_target,
)


def load_rgb(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    rgb = np.array(image).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(2, 0, 1)
    return rgb


def save_gray(l_tensor, save_path):
    gray = (l_tensor.squeeze(0).numpy() + 1.0) / 2.0
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(gray).save(save_path)


def save_rgb(rgb_tensor, save_path):
    rgb = rgb_tensor.permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="demo_output")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(args.image, args.size)
    l, ab = prepare_input_target(rgb)

    lab = merge_l_ab(l, ab)
    rgb_back = lab_to_rgb(denormalize_lab(lab)).squeeze(0)

    save_gray(l.squeeze(0), save_dir / "input_L.png")
    save_rgb(rgb_back, save_dir / "recover_rgb.png")

    print("L shape:", tuple(l.shape))
    print("ab shape:", tuple(ab.shape))
    print("灰度输入已保存到:", save_dir / "input_L.png")
    print("恢复彩色图已保存到:", save_dir / "recover_rgb.png")


if __name__ == "__main__":
    main()
