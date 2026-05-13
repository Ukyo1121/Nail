"""
单独测试 band ROI 对 nipple 分类效果的脚本。
对比两种模式：
  --mode original  nipple 使用原图（与 venous/arrangement/base_transparency 共享 backbone）
  --mode band      nipple 使用 band 裁剪（独立 backbone）

训练策略与 train_efficientnet.py 完全一致（ordinal CE + regression + AMP + EarlyStopping）。
"""

import os
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

from shape_tubes_dataset import ClassificationDataset


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def make_ordinal_soft_targets(labels, num_classes, sigma=1.0):
    device = labels.device
    mask = labels != -1
    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    classes = torch.arange(num_classes, device=device).float()
    safe_labels_float = safe_labels.float().unsqueeze(1)
    dist_sq = (classes.unsqueeze(0) - safe_labels_float) ** 2
    soft_targets = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
    return soft_targets, mask


def ordinal_ce_loss(logits, labels, num_classes, sigma=1.0):
    soft_targets, mask = make_ordinal_soft_targets(labels, num_classes, sigma)
    log_probs = F.log_softmax(logits, dim=1)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)
    if mask.sum() == 0:
        return per_sample_loss.sum() * 0.0
    return per_sample_loss[mask].mean()


def calculate_correct(logits, labels):
    _, predicted_classes = torch.max(logits, 1)
    mask = labels != -1
    correct = (predicted_classes[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


class NippleOnlyModel(nn.Module):
    """只保留 nipple 分类头（分类 + 回归）的模型。"""
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


def train_one_epoch(model, loader, optimizer, device, scaler, lambda_reg, sigma, use_band=False):
    model.train()
    running_loss = 0.0

    for images, band_images, labels in tqdm(loader, desc="Train"):
        nipple_labels = labels['nipple'].to(device)
        valid = nipple_labels != -1
        if valid.sum() == 0:
            continue

        input_img = band_images.to(device) if use_band else images.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits, reg_pred = model(input_img)
            ce_loss = ordinal_ce_loss(logits, nipple_labels, num_classes=4, sigma=sigma)
            mask = nipple_labels != -1
            if mask.sum() > 0:
                targets = nipple_labels[mask].float() / 3.0
                reg_loss = F.smooth_l1_loss(reg_pred[mask].squeeze(-1), targets)
            else:
                reg_loss = reg_pred.sum() * 0.0
            loss = ce_loss + lambda_reg * reg_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate_epoch(model, loader, device, lambda_reg, sigma, use_band=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = {}
    class_total = {}

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, band_images, labels in tqdm(loader, desc="Val"):
            nipple_labels = labels['nipple'].to(device)
            valid = nipple_labels != -1
            if valid.sum() == 0:
                continue

            input_img = band_images.to(device) if use_band else images.to(device)
            logits, reg_pred = model(input_img)

            ce_loss = ordinal_ce_loss(logits, nipple_labels, num_classes=4, sigma=sigma)
            mask = nipple_labels != -1
            if mask.sum() > 0:
                targets = nipple_labels[mask].float() / 3.0
                reg_loss = F.smooth_l1_loss(reg_pred[mask].squeeze(-1), targets)
            else:
                reg_loss = reg_pred.sum() * 0.0
            val_loss += (ce_loss + lambda_reg * reg_loss).item()

            c, n = calculate_correct(logits, nipple_labels)
            correct += c
            total += n

            _, preds = torch.max(logits, 1)
            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(nipple_labels[mask].cpu().numpy())

            for p, t in zip(preds[mask].cpu().tolist(), nipple_labels[mask].cpu().tolist()):
                class_total[t] = class_total.get(t, 0) + 1
                if p == t:
                    class_correct[t] = class_correct.get(t, 0) + 1

    avg_val_loss = val_loss / max(len(loader), 1)
    acc = correct / max(total, 1)

    preds_all = np.concatenate(all_preds) if all_preds else np.array([])
    labels_all = np.concatenate(all_labels) if all_labels else np.array([])
    if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
        macro_f1 = f1_score(labels_all, preds_all, average='macro')
        qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
    else:
        macro_f1 = 0.0
        qwk = 0.0

    per_class_acc = {}
    for c in sorted(class_total.keys()):
        per_class_acc[c] = class_correct.get(c, 0) / class_total[c]

    metrics = {
        'acc': acc, 'f1': macro_f1, 'qwk': qwk,
        'per_class_acc': per_class_acc, 'total': total
    }
    return avg_val_loss, metrics


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    IMG_SIZE = args.img_size

    train_transform = A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_band = os.path.join(args.band_dir, "train_band") if args.band_dir else None
    val_band = os.path.join(args.band_dir, "val_band") if args.band_dir else None
    use_band = (args.mode == "band")

    train_dataset = ClassificationDataset(
        annotation=args.train_ann, root=args.train_dir,
        transform=train_transform, band_dir=train_band
    )
    val_dataset = ClassificationDataset(
        annotation=args.val_ann, root=args.val_dir,
        transform=val_transform, band_dir=val_band
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = NippleOnlyModel(num_classes=4, pretrained=True).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=args.patience)

    logging.info(f"Mode: {args.mode} | IMG_SIZE={IMG_SIZE} | Band dir: {args.band_dir or 'N/A'} | Use band: {use_band}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.device, scaler,
            lambda_reg=args.lambda_reg, sigma=args.sigma, use_band=use_band
        )
        scheduler.step()
        logging.info(f"Epoch [{epoch + 1}/{args.epochs}] - Training Loss: {train_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loss, metrics = evaluate_epoch(
                model, val_loader, args.device,
                lambda_reg=args.lambda_reg, sigma=args.sigma, use_band=use_band
            )
            per_class_str = "  ".join(
                f"C{c}={metrics['per_class_acc'].get(c, 0):.4f}"
                for c in sorted(metrics['per_class_acc'].keys())
            )
            logging.info(
                f"Epoch [{epoch + 1}/{args.epochs}] - Validation Loss: {val_loss:.4f} | "
                f"Nipple: Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | QWK: {metrics['qwk']:.4f} | "
                f"Per-class: [{per_class_str}]"
            )

            if metrics['acc'] > best_val_acc:
                best_val_acc = metrics['acc']
                best_model_path = os.path.join(args.output_dir, f"nipple_{args.mode}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Best Val Acc: {best_val_acc:.4f} -> saved {best_model_path}")

            if early_stopping.step(metrics['acc']):
                logging.info(f"Early stopping at epoch {epoch + 1}, best Val Acc: {early_stopping.best_score:.4f}")
                break

        last_model_path = os.path.join(args.output_dir, f"nipple_{args.mode}_last.pth")
        torch.save(model.state_dict(), last_model_path)
        logging.info(f"Saved model of epoch {epoch + 1} to {last_model_path}")

    logging.info(f"Done. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="band", choices=["original", "band"],
                        help="original: nipple 用原图; band: nipple 用 band 裁剪")
    parser.add_argument("--train_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/train_classification.json")
    parser.add_argument("--train_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/train")
    parser.add_argument("--val_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json")
    parser.add_argument("--val_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val")
    parser.add_argument("--band_dir", type=str, default="/home/suzhiling/efficientnet/bands", help="band 裁剪根目录，下含 train_band/ val_band/")
    parser.add_argument("--output_dir", type=str, default="./work_dir/models/nipple_test_band/V2_1024")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_freq", type=int, default=4, help="Evaluate every N epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (eval cycles)")
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian sigma for ordinal soft labels")
    parser.add_argument("--lambda_reg", type=float, default=0.3, help="Weight for regression loss")
    parser.add_argument("--img_size", type=int, default=1024, help="Input image size (square)")
    args = parser.parse_args()
    main(args)
