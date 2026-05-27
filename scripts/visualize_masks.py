"""
可视化脚本：将每张图的血管 mask 叠加到原图上，并标注最高点。
用法：
    python visualize_masks.py --input_dir /path/to/images --mask_dir /path/to/masks --output_dir ./mask_viz
"""

import os
import argparse
import glob

import cv2
import numpy as np


def find_highest_point(mask):
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return None, None
    y_min = int(ys.min())
    xs_at_top = np.where(mask[y_min] > 0)[0]
    x_center = int(np.mean(xs_at_top))
    return x_center, y_min


def visualize_masks_for_image(image_path, mask_dir, save_path, alpha=0.4):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  无法读取图像: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    basename = os.path.splitext(os.path.basename(image_path))[0]
    mask_pattern = f"{basename}_mask_*.npy"
    mask_files = sorted(glob.glob(os.path.join(mask_dir, mask_pattern)))

    if not mask_files:
        print(f"  未找到 mask: {basename}, 跳过")
        return

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    # 左侧：原图 + mask 叠加；右侧：原图 + mask 叠加 + 最高点标记
    overlay = image_rgb.copy()
    overlay_with_points = image_rgb.copy()

    highest_points = []

    for idx, mf in enumerate(mask_files):
        mask_raw = np.load(mf)
        if mask_raw.shape[:2] != (h, w):
            mask = cv2.resize(
                mask_raw.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask = mask_raw
        color = colors[idx % len(colors)]
        mask_bool = mask > 0

        # 画 mask 区域
        color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        color_mask[mask_bool] = color
        overlay = np.where(
            mask_bool[:, :, None],
            (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8),
            overlay
        )
        overlay_with_points = np.where(
            mask_bool[:, :, None],
            (overlay_with_points * (1 - alpha) + color_mask * alpha).astype(np.uint8),
            overlay_with_points
        )

        # 画 mask 轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_with_points, contours, -1, color, 2)

        # 找并标注最高点
        x_top, y_top = find_highest_point(mask)
        if x_top is not None:
            highest_points.append((x_top, y_top, color, os.path.basename(mf)))

    # 在 overlay_with_points 上画最高点
    for x_top, y_top, color, mf_name in highest_points:
        cv2.circle(overlay_with_points, (x_top, y_top), 8, color, -1)
        cv2.circle(overlay_with_points, (x_top, y_top), 10, (255, 255, 255), 2)
        cv2.putText(
            overlay_with_points,
            f"({x_top},{y_top})",
            (x_top + 12, y_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    # 如果找到最高点，画裁剪范围线
    if highest_points:
        all_y = [p[1] for p in highest_points]
        y_top_bound = min(all_y)
        y_bottom_bound = max(all_y)
        cv2.line(overlay_with_points, (0, y_top_bound), (w - 1, y_top_bound), (0, 255, 255), 2)
        cv2.line(overlay_with_points, (0, y_bottom_bound), (w - 1, y_bottom_bound), (0, 255, 255), 2)
        cv2.putText(
            overlay_with_points,
            f"y_min(topmost tip)={y_top_bound}",
            (5, y_top_bound - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        cv2.putText(
            overlay_with_points,
            f"y_max(lowest tip)={y_bottom_bound}",
            (5, y_bottom_bound - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )

    # 拼接：左边原图+mask，右边标注版
    combined = np.hstack([overlay, overlay_with_points])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    info_text = (
        f"图像: {basename}  |  mask 数量: {len(mask_files)}  |  "
        f"最高点 y: {[p[1] for p in highest_points]}  |  "
        f"y_min(tips): {min(p[1] for p in highest_points) if highest_points else 'N/A'}  |  "
        f"y_max(tips): {max(p[1] for p in highest_points) if highest_points else 'N/A'}"
    )
    print(f"  {info_text}")


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

    # 先扫描所有 mask 文件，看看哪些图像有 mask
    all_masks = glob.glob(os.path.join(args.mask_dir, "*.npy"))
    mask_basenames = set()
    for mf in all_masks:
        fname = os.path.basename(mf)
        # 从 {file_name}_mask_{instance_id}.npy 中提取 file_name
        idx = fname.rfind("_mask_")
        if idx > 0:
            mask_basenames.add(fname[:idx])

    print(f"mask 目录中共 {len(all_masks)} 个 .npy 文件，覆盖 {len(mask_basenames)} 张图像")
    print()

    processed = 0
    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        if basename not in mask_basenames:
            continue
        save_path = os.path.join(args.output_dir, f"{basename}_mask_viz.jpg")
        print(f"[{i + 1}/{len(image_paths)}] {os.path.basename(img_path)}")
        visualize_masks_for_image(img_path, args.mask_dir, save_path)
        processed += 1

    if processed == 0:
        print("没有找到任何有 mask 的图像。"
              "请检查 mask 命名是否符合 {file_name}_mask_*.npy 格式。")
    else:
        print(f"\n全部完成，共处理 {processed} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val', help="原始图像目录")
    parser.add_argument("--mask_dir", type=str, default='/home/suzhiling/mask_rcnn_v2/work_dir/v2/0520/predication_v2_val/masks', help="mask .npy 文件目录")
    parser.add_argument("--output_dir", type=str, default="./mask_viz", help="可视化输出目录")
    args = parser.parse_args()
    main(args)
