"""
对比 original 和 band 两种模式下 nipple 分类的准确率。
用法：
    python eval_nipple_compare.py --original_model path/to/nipple_original_best.pth --band_model path/to/nipple_band_best.pth

评估策略与 eval_efficientnet.py 一致（Acc / F1 / QWK / per-class Recall）。
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, cohen_kappa_score

from shape_tubes_dataset import ClassificationDataset


class NippleOnlyModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)
        self.reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat), self.reg(feat)


@torch.no_grad()
def evaluate(model, loader, device, use_band=False):
    model.eval()
    correct = 0
    total = 0
    task_tp = {}
    task_fn = {}
    all_preds = []
    all_labels = []

    for images, band_images, labels in tqdm(loader, desc="Eval"):
        nipple_labels = labels['nipple'].to(device)
        valid = nipple_labels != -1
        if valid.sum() == 0:
            continue

        input_img = band_images.to(device) if use_band else images.to(device)
        logits, _ = model(input_img)
        _, preds = torch.max(logits, 1)

        mask = valid
        correct += (preds[mask] == nipple_labels[mask]).sum().item()
        total += mask.sum().item()

        all_preds.append(preds[mask].cpu().numpy())
        all_labels.append(nipple_labels[mask].cpu().numpy())

        for c in nipple_labels[mask].unique().tolist():
            c = int(c)
            gt_c = nipple_labels[mask] == c
            pred_c = preds[mask] == c
            tp = (gt_c & pred_c).sum().item()
            fn = (gt_c & ~pred_c).sum().item()
            task_tp[c] = task_tp.get(c, 0) + tp
            task_fn[c] = task_fn.get(c, 0) + fn

    acc = correct / max(total, 1)

    preds_all = np.concatenate(all_preds) if all_preds else np.array([])
    labels_all = np.concatenate(all_labels) if all_labels else np.array([])

    if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
        macro_f1 = f1_score(labels_all, preds_all, average='macro')
        qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
    else:
        macro_f1 = 0.0
        qwk = 0.0

    all_classes = sorted(set(task_tp.keys()) | set(task_fn.keys()))
    class_recalls = {}
    for c in all_classes:
        tp = task_tp.get(c, 0)
        fn = task_fn.get(c, 0)
        class_recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_recall = sum(class_recalls.values()) / len(class_recalls) if class_recalls else 0.0

    metrics = {
        'acc': acc,
        'f1': macro_f1,
        'qwk': qwk,
        'recall': macro_recall,
        'class_recalls': class_recalls,
        'total': total,
        'correct': correct,
    }
    return metrics


def main(args):
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    band_dir = args.band_dir
    val_dataset = ClassificationDataset(
        annotation=args.val_ann, root=args.val_dir,
        transform=val_transform, band_dir=band_dir
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = args.device

    # --- 加载模型 ---
    print("\n=== Loading Models ===")
    model_orig = NippleOnlyModel(num_classes=4, pretrained=False).to(device)
    model_orig.load_state_dict(torch.load(args.original_model, map_location=device))
    model_band = NippleOnlyModel(num_classes=4, pretrained=False).to(device)
    model_band.load_state_dict(torch.load(args.band_model, map_location=device))

    # --- 四种组合评估 ---
    configs = [
        ("Orig Model + Original Img", model_orig, False),
        ("Orig Model + Band Img",     model_orig, True),
        ("Band Model + Original Img", model_band, False),
        ("Band Model + Band Img",     model_band, True),
    ]

    all_results = {}
    lines = []

    for name, model, use_band in configs:
        print(f"\n=== {name} ===")
        metrics = evaluate(model, val_loader, device, use_band=use_band)
        all_results[name] = metrics

        line = (f"  {name:>35s}  Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | "
                f"QWK: {metrics['qwk']:.4f} | Recall(macro): {metrics['recall']:.4f}  "
                f"({metrics['correct']}/{metrics['total']})")
        print(line)
        lines.append(line)

        recall_strs = [f"C{c}={metrics['class_recalls'][c]:.4f}" for c in sorted(metrics['class_recalls'].keys())]
        rec_line = f"    Per-class Recall: [{', '.join(recall_strs)}]"
        print(rec_line)
        lines.append(rec_line)

    # --- 汇总对比 ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    lines.append("\n" + "=" * 60)
    lines.append("Summary")
    lines.append("=" * 60)

    header = f"  {'Metric':<16}" + "".join(f"{name:>23}" for name, _, _ in configs)
    print(f"\n{header}")
    lines.append(header)

    print(f"  {'-'*16}" + "".join(f"{'-'*23}" for _ in configs))
    for metric_key, metric_label in [('acc', 'Acc'), ('f1', 'F1'), ('qwk', 'QWK'), ('recall', 'Recall(macro)')]:
        row = f"  {metric_label:<16}"
        for name, _, _ in configs:
            row += f"{all_results[name][metric_key]:>23.4f}"
        print(row)
        lines.append(row)

    # 每类 Recall 对比表
    all_classes = sorted(set().union(*(r['class_recalls'].keys() for r in all_results.values())))
    if all_classes:
        print(f"\n  {'Class':<8}" + "".join(f"{name:>23}" for name, _, _ in configs))
        print(f"  {'-'*8}" + "".join(f"{'-'*23}" for _ in configs))
        for c in all_classes:
            row = f"  C{c:<7}"
            for name, _, _ in configs:
                val = all_results[name]['class_recalls'].get(c, 0)
                row += f"{val:>23.4f}"
            print(row)
            lines.append(row)

    # --- 保存日志 ---
    for model_path in [args.original_model, args.band_model]:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        log_path = os.path.join(model_dir, 'eval_logs.txt')
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, 'a') as f:
            f.write(f"{'='*60}\n[{timestamp}] Eval nipples compare\n")
            f.write(f"  original_model: {args.original_model}\n")
            f.write(f"  band_model:     {args.band_model}\n")
            f.write(f"{'='*60}\n")
            f.write('\n'.join(lines) + '\n\n')
        print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", type=str, default="./work_dir/models/nipple_test_original/V2_1024/nipple_original_best.pth")
    parser.add_argument("--band_model", type=str, default="./work_dir/models/nipple_test_band/V2_1024/nipple_band_best.pth")
    parser.add_argument("--val_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json")
    parser.add_argument("--val_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val")
    parser.add_argument("--band_dir", type=str, default="/home/suzhiling/efficientnet/bands/val_band", help="band 裁剪目录（val）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=1024, help="Input image size (square)")
    args = parser.parse_args()
    main(args)
