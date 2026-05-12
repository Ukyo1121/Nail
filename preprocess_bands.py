"""
预处理脚本：对文件夹中所有图像提取边界带状区域并裁剪保存。
用法：
    python preprocess_bands.py --input_dir /path/to/images --output_dir /path/to/band_images
"""

import os
import argparse
import glob

from edge_extract import extract_and_save_band


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"未找到图像: {args.input_dir}")
        return

    print(f"共找到 {len(image_paths)} 张图像")
    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(args.output_dir, f"{basename}_band.jpg")
        print(f"[{i + 1}/{len(image_paths)}] {os.path.basename(img_path)}")
        try:
            extract_and_save_band(img_path, save_path, band_width=args.band_width)
        except Exception as e:
            print(f"  跳过 {img_path}: {e}")

    print("全部完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val', help="原始图像目录")
    parser.add_argument("--output_dir", type=str, default='/home/suzhiling/efficientnet/bands/val_band', help="band 裁剪输出目录")
    parser.add_argument("--band_width", type=int, default=200, help="带状区域半宽（像素）")
    args = parser.parse_args()
    main(args)
