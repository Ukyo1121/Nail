import os
import cv2
import numpy as np


def normalize_to_uint8(x):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return (x * 255).astype(np.uint8)


def smooth_1d(y, win=21):
    """
    简单滑动平均平滑曲线
    """
    if win % 2 == 0:
        win += 1
    pad = win // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth


def build_boundary_score(image_rgb):
    """
    构建边界可能性 score。
    score 越大，越可能是甲襞上缘边界。

    核心思想：
    1. 不用单纯局部梯度；
    2. 比较上方窗口和下方窗口的颜色/亮度差异；
    3. 抑制强反光；
    4. 抑制红色血管区域；
    5. 加入位置先验。
    """

    h, w = image_rgb.shape[:2]

    # RGB -> LAB，L 表示亮度，A/B 表示颜色
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # 轻微平滑，降低噪声
    L_blur = cv2.GaussianBlur(L, (9, 9), 0)
    A_blur = cv2.GaussianBlur(A, (9, 9), 0)
    B_blur = cv2.GaussianBlur(B, (9, 9), 0)

    # 用上下窗口差异构建“边界感”
    # window 越大，越关注区域差异，而不是细小边缘
    offset = max(8, h // 60)

    score = np.zeros((h, w), dtype=np.float32)

    for y in range(offset, h - offset):
        upper_L = L_blur[y - offset, :]
        lower_L = L_blur[y + offset, :]

        upper_A = A_blur[y - offset, :]
        lower_A = A_blur[y + offset, :]

        upper_B = B_blur[y - offset, :]
        lower_B = B_blur[y + offset, :]

        # 上下区域差异，不只看亮度，也看颜色
        diff_L = np.abs(lower_L - upper_L)
        diff_A = np.abs(lower_A - upper_A)
        diff_B = np.abs(lower_B - upper_B)

        score[y, :] = 0.5 * diff_L + 0.25 * diff_A + 0.25 * diff_B

    # 归一化
    score = score - score.min()
    score = score / (score.max() + 1e-6)

    # ---------- 抑制强反光 ----------
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    highlight = (gray > 220).astype(np.float32)
    highlight = cv2.dilate(highlight, np.ones((9, 9), np.uint8), iterations=1)

    # 反光区域虽然边缘强，但不希望路径走那里
    score = score * (1.0 - 0.75 * highlight)

    # ---------- 抑制红色血管 ----------
    R = image_rgb[:, :, 0].astype(np.float32)
    G = image_rgb[:, :, 1].astype(np.float32)
    Bc = image_rgb[:, :, 2].astype(np.float32)

    # 红色显著区域：R 明显大于 G/B
    redness = R - 0.5 * G - 0.5 * Bc
    redness = redness - redness.min()
    redness = redness / (redness.max() + 1e-6)

    vessel_like = (redness > 0.55).astype(np.float32)
    vessel_like = cv2.dilate(vessel_like, np.ones((5, 5), np.uint8), iterations=1)

    # 血管区域也降低 score
    score = score * (1.0 - 0.7 * vessel_like)

    # ---------- 位置软先验 ----------
    # 不是硬裁剪，只是让中上部更可能成为边界
    yy = np.linspace(0, 1, h).reshape(h, 1)

    # 你可以根据数据调整 center
    # 如果边界通常在图像高度 20%~45%，center=0.32 比较合适
    center = 0.32
    sigma = 0.22
    pos_prior = np.exp(-((yy - center) ** 2) / (2 * sigma ** 2))

    score = score * pos_prior

    # 再归一化
    score = score - score.min()
    score = score / (score.max() + 1e-6)

    return score


def dynamic_programming_path(score, smooth_penalty=4.0, max_jump=8):
    """
    在 score map 上找一条从左到右的最优路径。

    score 越大越好。
    动态规划中转成 cost = -score。

    smooth_penalty 控制曲线平滑程度：
        越大，线越平滑，不容易上下乱跳。
    max_jump 控制相邻列 y 坐标最大跳动范围。
    """

    h, w = score.shape

    cost = -score.astype(np.float32)

    dp = np.full((h, w), np.inf, dtype=np.float32)
    backtrack = np.zeros((h, w), dtype=np.int32)

    dp[:, 0] = cost[:, 0]

    for x in range(1, w):
        for y in range(h):
            y_min = max(0, y - max_jump)
            y_max = min(h, y + max_jump + 1)

            prev_ys = np.arange(y_min, y_max)

            # 平滑惩罚：相邻列跳动越大，代价越高
            jump_cost = smooth_penalty * ((prev_ys - y) ** 2) / (max_jump ** 2)

            candidates = dp[y_min:y_max, x - 1] + jump_cost

            best_idx = np.argmin(candidates)
            best_prev_y = prev_ys[best_idx]

            dp[y, x] = cost[y, x] + candidates[best_idx]
            backtrack[y, x] = best_prev_y

    # 回溯
    path = np.zeros(w, dtype=np.int32)
    path[-1] = np.argmin(dp[:, -1])

    for x in range(w - 1, 0, -1):
        path[x - 1] = backtrack[path[x], x]

    return path


def extract_boundary_line(image_rgb, resize_width=768):
    """
    输入 RGB 图像，输出边界线坐标。
    为了速度和稳定性，先 resize 到固定宽度处理。
    """

    h0, w0 = image_rgb.shape[:2]

    scale = resize_width / w0
    resize_height = int(h0 * scale)

    image_resized = cv2.resize(
        image_rgb,
        (resize_width, resize_height),
        interpolation=cv2.INTER_LINEAR
    )

    score = build_boundary_score(image_resized)

    path = dynamic_programming_path(
        score,
        smooth_penalty=5.0,
        max_jump=max(4, resize_height // 80)
    )

    # 平滑曲线
    path_smooth = smooth_1d(path.astype(np.float32), win=31)

    # 映射回原图坐标
    xs_resized = np.arange(resize_width)
    xs_original = xs_resized / scale
    ys_original = path_smooth / scale

    return xs_original, ys_original, score, image_resized, path_smooth


def draw_boundary_overlay(image_rgb, xs, ys, color=(255, 0, 0), thickness=3):
    """
    在原图上画边界线。
    color 是 RGB。
    """
    overlay = image_rgb.copy()
    pts = []

    h, w = image_rgb.shape[:2]

    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))

        if 0 <= xi < w and 0 <= yi < h:
            pts.append([xi, yi])

    pts = np.array(pts, dtype=np.int32)

    if len(pts) > 1:
        cv2.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=color,
            thickness=thickness
        )

    return overlay


def draw_band_overlay(image_rgb, xs, ys, band_width=120):
    """
    画出沿边界线上下扩展的带状区域。
    这个 band 后续可以作为 ROI 使用。
    """
    overlay = image_rgb.copy()
    h, w = image_rgb.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    upper_pts = []
    lower_pts = []

    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))

        if 0 <= xi < w:
            y1 = max(0, yi - band_width)
            y2 = min(h - 1, yi + band_width)
            upper_pts.append([xi, y1])
            lower_pts.append([xi, y2])

    if len(upper_pts) > 1:
        polygon = np.array(upper_pts + lower_pts[::-1], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

    # 红色透明覆盖
    color_mask = np.zeros_like(image_rgb)
    color_mask[:, :, 0] = 255

    alpha = 0.25
    overlay = np.where(
        mask[:, :, None] > 0,
        (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8),
        overlay
    )

    # 画中心线
    overlay = draw_boundary_overlay(overlay, xs, ys, color=(255, 0, 0), thickness=3)

    return overlay, mask


def debug_boundary_extraction(image_path, save_dir="./boundary_debug"):
    os.makedirs(save_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    xs, ys, score, image_resized, path_resized = extract_boundary_line(
        image_rgb,
        resize_width=768
    )

    line_overlay = draw_boundary_overlay(
        image_rgb,
        xs,
        ys,
        color=(255, 0, 0),
        thickness=3
    )

    band_overlay, band_mask = draw_band_overlay(
        image_rgb,
        xs,
        ys,
        band_width=200
    )

    base = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(
        os.path.join(save_dir, f"{base}_00_original.jpg"),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(save_dir, f"{base}_01_score_map.jpg"),
        normalize_to_uint8(score)
    )

    cv2.imwrite(
        os.path.join(save_dir, f"{base}_02_boundary_line.jpg"),
        cv2.cvtColor(line_overlay, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(save_dir, f"{base}_03_boundary_band.jpg"),
        cv2.cvtColor(band_overlay, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(save_dir, f"{base}_04_band_mask.jpg"),
        band_mask
    )

    # 保存边界坐标
    coords = np.stack([xs, ys], axis=1)
    np.savetxt(
        os.path.join(save_dir, f"{base}_boundary_coords.txt"),
        coords,
        fmt="%.2f",
        delimiter=",",
        header="x,y"
    )


def extract_and_save_band(image_path, save_path, band_width=120):
    """
    提取边界带状区域并裁剪保存。
    返回裁剪图像（RGB），同时写入 save_path。
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    xs, ys, _, _, _ = extract_boundary_line(image_rgb, resize_width=768)
    h, w = image_rgb.shape[:2]

    # 计算 band 多边形的包围盒
    upper_ys = []
    lower_ys = []
    for x, y in zip(xs, ys):
        xi = int(round(x))
        if 0 <= xi < w:
            yi = int(round(y))
            upper_ys.append(max(0, yi - band_width))
            lower_ys.append(min(h - 1, yi + band_width))

    if not upper_ys:
        # 边界提取失败，保存原图
        cv2.imwrite(save_path, image_bgr)
        return image_rgb

    y_min = min(upper_ys)
    y_max = max(lower_ys)

    crop = image_rgb[y_min:y_max + 1, 0:w]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return crop


if __name__ == "__main__":
    import glob

    input_dir = "/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val"
    save_dir = "./edge_test_img"

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"未找到图像: {input_dir}")
    else:
        print(f"共找到 {len(image_paths)} 张图像")
        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")
            debug_boundary_extraction(img_path, save_dir=save_dir)
        print("全部完成")
