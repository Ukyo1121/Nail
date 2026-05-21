"""计算 bands_v2 下所有裁剪图片的宽高比统计。"""
import os
from PIL import Image
import numpy as np

base = "/home/suzhiling/efficientnet/bands_v2"

all_ratios = []
for sub in ["train_band", "val_band"]:
    subdir = os.path.join(base, sub)
    if not os.path.isdir(subdir):
        continue
    for fname in sorted(os.listdir(subdir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(subdir, fname)
        try:
            with Image.open(path) as img:
                w, h = img.size
                ratio = w / h  # 宽高比
                all_ratios.append((fname, w, h, ratio))
        except Exception as e:
            print(f"Error reading {path}: {e}")

ratios = np.array([r[3] for r in all_ratios])
widths = np.array([r[1] for r in all_ratios])
heights = np.array([r[2] for r in all_ratios])

print(f"总图片数: {len(all_ratios)}")
print(f"\n宽高比 (W/H):")
print(f"  mean:  {ratios.mean():.4f}")
print(f"  std:   {ratios.std():.4f}")
print(f"  min:   {ratios.min():.4f}")
print(f"  max:   {ratios.max():.4f}")
print(f"  median:{np.median(ratios):.4f}")
print(f"  p5:    {np.percentile(ratios, 5):.4f}")
print(f"  p95:   {np.percentile(ratios, 95):.4f}")

print(f"\n宽度统计:")
print(f"  mean: {widths.mean():.1f}, std: {widths.std():.1f}, min: {widths.min()}, max: {widths.max()}")
print(f"  p5: {np.percentile(widths, 5):.1f}, p50: {np.percentile(widths, 50):.1f}, p95: {np.percentile(widths, 95):.1f}")

print(f"\n高度统计:")
print(f"  mean: {heights.mean():.1f}, std: {heights.std():.1f}, min: {heights.min()}, max: {heights.max()}")
print(f"  p5: {np.percentile(heights, 5):.1f}, p50: {np.percentile(heights, 50):.1f}, p95: {np.percentile(heights, 95):.1f}")

# 宽高比分布
print("\n宽高比分布直方图:")
bins = [0, 3, 4, 5, 6, 7, 8, 9, 10, 15, 50]
hist, _ = np.histogram(ratios, bins=bins)
for i in range(len(hist)):
    print(f"  [{bins[i]:>3}, {bins[i+1]:>3}): {hist[i]}")

# 如果宽高比稳定在 4:1 附近，给出推荐的 resize 尺寸
print(f"\n推荐固定 resize 尺寸:")
median_h = int(np.median(heights))
median_w = int(np.median(widths))
print(f"  median 尺寸: {median_w}×{median_h}, 宽高比 {median_w/median_h:.2f}")

# 给出两个选项
print(f"  方案 A (4:1): Resize((384, 96)) 或 Resize((512, 128)) 或 Resize((768, 192))")
print(f"  方案 B (接近 median ratio): Resize(({int(median_h * 4)}, {median_h})) 更贴合实际比例")
