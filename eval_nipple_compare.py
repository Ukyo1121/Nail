"""
对比 original 和 band 两种模式下 nipple 分类的准确率。
用法：
    python eval_nipple_compare.py --original_model path/to/nipple_original_best.pth --band_model path/to/nipple_band_best.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
from tqdm import tqdm
import argparse
from sklearn.metrics import cohen_kappa_score

from shape_tubes_dataset import ClassificationDataset


class NippleOnlyModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.0):
        super().__init__()
        self.backbone = mobilenet_v3_large(pretrained=pretrained)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)
        self.reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.fc(feat), self.reg(feat)


@torch.no_grad()
def evaluate(model, loader, device, use_band=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, band_images, labels in tqdm(loader, desc="Eval"):
        nipple_labels = labels['nipple'].to(device)
        valid = nipple_labels != -1
        if valid.sum() == 0:
            continue

        input_img = band_images.to(device) if use_band else images.to(device)
        logits, _ = model(input_img)
        preds = logits[valid].argmax(dim=1)
        targets = nipple_labels[valid]

        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    acc = correct / max(total, 1)

    # 每类统计
    class_actual = {}   # 实际属于该类的样本数
    class_pred = {}     # 预测为该类的样本数
    class_tp = {}       # 正确预测为该类的样本数
    for p, t in zip(all_preds, all_targets):
        class_actual[t] = class_actual.get(t, 0) + 1
        class_pred[p] = class_pred.get(p, 0) + 1
        if p == t:
            class_tp[t] = class_tp.get(t, 0) + 1

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    for c in sorted(class_actual.keys()):
        tp = class_tp.get(c, 0)
        precision = tp / class_pred.get(c, 1)
        recall = tp / class_actual[c]
        per_class_precision[c] = precision
        per_class_recall[c] = recall
        per_class_f1[c] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    macro_precision = sum(per_class_precision.values()) / max(len(per_class_precision), 1)
    macro_recall = sum(per_class_recall.values()) / max(len(per_class_recall), 1)
    macro_f1 = sum(per_class_f1.values()) / max(len(per_class_f1), 1)

    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic') if len(set(all_targets)) > 1 else 0.0

    return acc, per_class_precision, per_class_recall, per_class_f1, macro_precision, macro_recall, macro_f1, qwk, total


def main(args):
    val_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    band_dir = args.band_dir
    val_dataset = ClassificationDataset(
        annotation=args.val_ann, root=args.val_dir,
        transform=val_transform, band_dir=band_dir,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = args.device

    # --- 加载模型 ---
    print("\n=== Loading Models ===")
    model_orig = NippleOnlyModel(num_classes=4, pretrained=False).to(device)
    model_orig.load_state_dict(torch.load(args.original_model, map_location=device))
    model_band = NippleOnlyModel(num_classes=4, pretrained=False).to(device)
    model_band.load_state_dict(torch.load(args.band_model, map_location=device))

    # --- 两种模式评估 ---
    configs = [
        ("Original Model + Original Img", model_orig, False),
        ("Band Model + Band Img",         model_band, True),
    ]

    all_results = {}
    for name, model, use_band in configs:
        print(f"\n=== {name} ===")
        acc, cls_prec, cls_recall, cls_f1, macro_prec, macro_recall, macro_f1, qwk, total = evaluate(model, val_loader, device, use_band=use_band)
        all_results[name] = {
            'acc': acc, 'qwk': qwk, 'per_class_precision': cls_prec,
            'per_class_recall': cls_recall, 'per_class_f1': cls_f1,
            'macro_precision': macro_prec, 'macro_recall': macro_recall, 'macro_f1': macro_f1,
        }
        print(f"  Overall Acc: {acc:.4f}  ({int(acc * total)}/{total})  QWK: {qwk:.4f}")
        print(f"  Macro Precision: {macro_prec:.4f}  Macro Recall: {macro_recall:.4f}  Macro F1: {macro_f1:.4f}")
        for c in sorted(cls_prec.keys()):
            print(f"  Class {c}: Precision={cls_prec[c]:.4f}  Recall={cls_recall.get(c, 0):.4f}  F1={cls_f1.get(c, 0):.4f}")

    # --- 汇总对比 ---
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, res in all_results.items():
        print(f"  {name:<35}  Acc={res['acc']:.4f}  Macro F1={res['macro_f1']:.4f}  QWK={res['qwk']:.4f}")

    # 每类指标对比表
    all_classes = sorted(set().union(*(r['per_class_precision'].keys() for r in all_results.values())))
    for metric, key in [('Precision', 'per_class_precision'), ('Recall', 'per_class_recall'), ('F1', 'per_class_f1')]:
        header = f"\n  {'Per-Class ' + metric:<16}" + "".join(f"{name:>30}" for name, _, _ in configs)
        print(header)
        for c in all_classes:
            row = f"  {'Class ' + str(c):<16}"
            for name, _, _ in configs:
                val = all_results[name][key].get(c, 0)
                row += f"{val:>30.4f}"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", type=str, default="./work_dir/models/nipple_test_original/V7/nipple_original_best.pth")
    parser.add_argument("--band_model", type=str, default="./work_dir/models/nipple_test_band/V7/nipple_band_best.pth")
    parser.add_argument("--val_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json")
    parser.add_argument("--val_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val")
    parser.add_argument("--band_dir", type=str, default="/home/suzhiling/efficientnet/bands/v2/val_band", help="band 裁剪目录（val）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()
    main(args)
