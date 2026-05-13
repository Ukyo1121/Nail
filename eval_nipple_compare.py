"""
对比 original 和 band 两种模式下 nipple 分类的准确率。
用法：
    python eval_nipple_compare.py --original_model path/to/nipple_original_best.pth --band_model path/to/nipple_band_best.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from tqdm import tqdm
import argparse

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
    class_correct = {}
    class_total = {}

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

        # 每类统计
        for p, t in zip(preds.cpu().tolist(), targets.cpu().tolist()):
            class_total[t] = class_total.get(t, 0) + 1
            if p == t:
                class_correct[t] = class_correct.get(t, 0) + 1

    acc = correct / max(total, 1)
    per_class = {}
    for c in sorted(class_total.keys()):
        per_class[c] = class_correct.get(c, 0) / class_total[c]

    return acc, per_class, total


def main(args):
    val_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
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
    # orig模型+原图, orig模型+band, band模型+原图, band模型+band
    configs = [
        ("Orig Model + Original Img", model_orig, False),
        ("Orig Model + Band Img",     model_orig, True),
        ("Band Model + Original Img", model_band, False),
        ("Band Model + Band Img",     model_band, True),
    ]

    all_results = {}
    for name, model, use_band in configs:
        print(f"\n=== {name} ===")
        acc, cls_acc, total = evaluate(model, val_loader, device, use_band=use_band)
        all_results[name] = {'acc': acc, 'per_class': cls_acc}
        print(f"  Overall Acc: {acc:.4f}  ({int(acc * total)}/{total})")
        for c, a in cls_acc.items():
            print(f"  Class {c}: {a:.4f}")

    # --- 汇总对比 ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name:<35} {res['acc']:.4f}")

    # 每类对比表
    all_classes = sorted(set().union(*(r['per_class'].keys() for r in all_results.values())))
    header = f"  {'Class':<8}" + "".join(f"{name:>18}" for name, _, _ in configs)
    print(f"\n{header}")
    print(f"  {'-'*8}" + "".join(f"{'-'*18}" for _ in configs))
    for c in all_classes:
        row = f"  {c:<8}"
        for name, _, _ in configs:
            val = all_results[name]['per_class'].get(c, 0)
            row += f"{val:>18.4f}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", type=str, default="./work_dir/models/nipple_test_original/nipple_original_best.pth")
    parser.add_argument("--band_model", type=str, default="./work_dir/models/nipple_test_band/nipple_band_best.pth")
    parser.add_argument("--val_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json")
    parser.add_argument("--val_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val")
    parser.add_argument("--band_dir", type=str, default="/home/suzhiling/efficientnet/bands/val_band", help="band 裁剪目录（val）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
