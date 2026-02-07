
import argparse, os
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import torchvision.utils as vutils
import torch


def load_image(path=None):

        img = Image.open(path).convert("RGB")
        name = os.path.splitext(os.path.basename(path))[0]

        return img, name

def pil_to_tensor(img):
    return F.to_tensor(img)

def tensor_to_pil(t):
    return F.to_pil_image(t.clamp(0,1))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="philly.png", help="path to input image (RGB)")
    parser.add_argument("--out", type=str, default="aug_out", help="output directory")
    args = parser.parse_args()

    ensure_dir(args.out)
    img, base = load_image(args.img)

    W, H = img.size
    
    img_t = pil_to_tensor(img)

    results = []  

    # 1) ShearX
    sx = 15.0
    im = F.affine(img, angle=0.0, translate=(0,0), scale=1.0, shear=(sx, 0), interpolation=Image.BILINEAR, fill=0)
    results.append((f"ShearX(sx={sx}°)", im))

    # 2) ShearY
    sy = 15.0
    im = F.affine(img, angle=0.0, translate=(0,0), scale=1.0, shear=(0, sy), interpolation=Image.BILINEAR, fill=0)
    results.append((f"ShearY(sy={sy}°)", im))

    # 3) TranslateX
    tx = int(0.10 * W)
    im = F.affine(img, angle=0.0, translate=(tx, 0), scale=1.0, shear=(0.0, 0.0), interpolation=Image.BILINEAR, fill=0)
    results.append((f"TranslateX(tx={tx}px≈0.10W)", im))

    # 4) TranslateY
    ty = int(0.10 * H)
    im = F.affine(img, angle=0.0, translate=(0, ty), scale=1.0, shear=(0.0, 0.0), interpolation=Image.BILINEAR, fill=0)
    results.append((f"TranslateY(ty={ty}px≈0.10H)", im))

    # 5) Rotate
    angle = 30.0
    im = F.rotate(img, angle=angle, interpolation=Image.BILINEAR, fill=0)
    results.append((f"Rotate(angle={angle}°)", im))

    # 6) Brightness
    b_factor = 1.5
    im = F.adjust_brightness(img, b_factor)
    results.append((f"Brightness(factor={b_factor})", im))

    # 7) Color (Saturation)
    c_factor = 1.5
    im = F.adjust_saturation(img, c_factor)
    results.append((f"Color/Saturation(factor={c_factor})", im))

    # 8) Contrast
    ctr_factor = 1.5
    im = F.adjust_contrast(img, ctr_factor)
    results.append((f"Contrast(factor={ctr_factor})", im))

    # 9) Sharpness
    sh_factor = 2.0
    im = F.adjust_sharpness(img, sh_factor)
    results.append((f"Sharpness(factor={sh_factor})", im))

    # 10) Posterize
    bits = 4
    im = F.posterize(img, bits)
    results.append((f"Posterize(bits={bits})", im))

    # 11) Solarize
    thr = 128
    im = F.solarize(img, threshold=thr)
    results.append((f"Solarize(threshold={thr})", im))

    # 12) Equalize
    im = F.equalize(img)
    results.append(("Equalize()", im))


    grid_tensors = []
    orig_name = f"{base}_original.jpg"
    img.save(os.path.join(args.out, orig_name))
    grid_tensors.append(pil_to_tensor(img))
    print(f"Saved: {orig_name}")

    for title, im in results:
        fname = f"{base}_{title.replace('/','-').replace('°','deg').replace('≈','~').replace(' ','_').replace('(','[').replace(')',']')}.jpg"
        im.save(os.path.join(args.out, fname))
        print(f"Saved: {fname}")
        grid_tensors.append(pil_to_tensor(im))

    grid = vutils.make_grid(torch.stack(grid_tensors, dim=0), nrow=4, padding=4)
    grid_img = tensor_to_pil(grid)
    grid_path = os.path.join(args.out, f"{base}_augment_grid_4x3.jpg")
    grid_img.save(grid_path)
    print(f"Saved grid: {grid_path}")

    txt_path = os.path.join(args.out, f"{base}_augment_params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Augment Parameters\n")
        f.write(f"Original: {orig_name}\n")
        for title, _ in results:
            f.write(f"{title}\n")
    print(f"Wrote params: {txt_path}")
