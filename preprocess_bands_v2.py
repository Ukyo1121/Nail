"""
预处理脚本：基于血管 mask 最高点提取带状区域并裁剪保存。

对每张图像，找到其对应的所有血管 mask（{image_id}_mask_{instance_id}.npy），
提取每个 mask 的最高点，以所有最高点的最小值为基准向上扩展 top_offset 作为裁剪上界，
以所有最高点的最大值为裁剪下界。

用法：
    python preprocess_bands.py --input_dir /path/to/images --mask_dir /path/to/masks --output_dir /path/to/band_images
"""

import os
import argparse
import glob

from edge_extract import extract_and_save_band_from_masks


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
            extract_and_save_band_from_masks(
                img_path, save_path,
                mask_dir=args.mask_dir,
                top_offset=args.top_offset,
                bottom_offset=args.bottom_offset,
            )
        except Exception as e:
            print(f"  跳过 {img_path}: {e}")

    print("全部完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val', help="原始图像目录")
    parser.add_argument("--mask_dir", type=str, default='/home/suzhiling/mask_rcnn_v2/work_dir/v2/0520/predication_v2_val/masks', help="血管 mask .npy 文件目录")
    parser.add_argument("--output_dir", type=str, default='/home/suzhiling/efficientnet/bands_v2/val_band', help="band 裁剪输出目录")
    parser.add_argument("--top_offset", type=int, default=200, help="裁剪顶部向上扩展的像素数")
    parser.add_argument("--bottom_offset", type=int, default=200, help="裁剪底部向下扩展的像素数")
    args = parser.parse_args()
    main(args)
