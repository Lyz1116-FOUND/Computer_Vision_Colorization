import torch


def rgb_to_lab(rgb):
    """RGB 转 Lab，RGB 要在 0~1。"""
    one = False
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
        one = True

    rgb = torch.clamp(rgb, 0.0, 1.0)

    linear = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    r = linear[:, 0:1]
    g = linear[:, 1:2]
    b = linear[:, 2:3]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    def f(t):
        return torch.where(
            t > 0.008856,
            t ** (1.0 / 3.0),
            (903.3 * t + 16.0) / 116.0,
        )

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = torch.cat([l, a, b], dim=1)
    if one:
        lab = lab.squeeze(0)
    return lab


def lab_to_rgb(lab):
    """Lab 转回 RGB。"""
    one = False
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
        one = True

    l = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]

    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xr = torch.where(fx ** 3 > 0.008856, fx ** 3, (116.0 * fx - 16.0) / 903.3)
    yr = torch.where(fy ** 3 > 0.008856, fy ** 3, (116.0 * fy - 16.0) / 903.3)
    zr = torch.where(fz ** 3 > 0.008856, fz ** 3, (116.0 * fz - 16.0) / 903.3)

    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    rgb = torch.cat([r, g, b], dim=1)
    rgb = torch.where(
        rgb > 0.0031308,
        1.055 * (rgb ** (1.0 / 2.4)) - 0.055,
        12.92 * rgb,
    )
    rgb = torch.clamp(rgb, 0.0, 1.0)

    if one:
        rgb = rgb.squeeze(0)
    return rgb


def normalize_lab(lab):
    """Lab 归一化到常用范围。"""
    out = lab.clone()
    out[:, 0:1] = out[:, 0:1] / 50.0 - 1.0
    out[:, 1:3] = out[:, 1:3] / 128.0
    return out


def denormalize_lab(lab):
    """把归一化的 Lab 变回去。"""
    out = lab.clone()
    out[:, 0:1] = (out[:, 0:1] + 1.0) * 50.0
    out[:, 1:3] = out[:, 1:3] * 128.0
    return out


def split_l_ab(lab):
    """拆成 L 和 ab。"""
    return lab[:, 0:1], lab[:, 1:3]


def merge_l_ab(l, ab):
    """把 L 和 ab 拼回去。"""
    return torch.cat([l, ab], dim=1)


def prepare_input_target(rgb):
    """一张 RGB 图直接拆成输入和标签。"""
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    lab = rgb_to_lab(rgb)
    lab = normalize_lab(lab)
    l, ab = split_l_ab(lab)
    return l, ab
