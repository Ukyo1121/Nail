"""
单独测试 band ROI 对 nipple 分类效果的脚本。
对比两种模式：
  --mode original  nipple 使用原图（与 venous/arrangement/base_transparency 共享 backbone）
  --mode band      nipple 使用 band 裁剪（独立 backbone）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

from shape_tubes_dataset import ClassificationDataset


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


def make_ordinal_soft_targets(labels, num_classes, sigma=1.0):
    """为每个标签生成高斯软标签分布，距离越近权重越大。"""
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
    """Ordinal-aware cross entropy: 用高斯软标签替代 one-hot，让邻近类的惩罚更小。"""
    soft_targets, mask = make_ordinal_soft_targets(labels, num_classes, sigma)
    log_probs = F.log_softmax(logits, dim=1)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)
    if mask.sum() == 0:
        return per_sample_loss.sum() * 0.0
    return per_sample_loss[mask].mean()


def train_one_epoch(model, loader, optimizer, device, use_band=False, sigma=1.0, lambda_reg=0.3):
    model.train()
    reg_fn = nn.SmoothL1Loss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, band_images, labels in tqdm(loader, desc="Train"):
        nipple_labels = labels['nipple'].to(device)
        valid = nipple_labels != -1
        if valid.sum() == 0:
            continue

        input_img = band_images.to(device) if use_band else images.to(device)

        optimizer.zero_grad()
        logits, reg_pred = model(input_img)

        loss_ce = ordinal_ce_loss(logits[valid], nipple_labels[valid], num_classes=4, sigma=sigma)
        reg_targets = nipple_labels[valid].float() / 3.0
        loss_reg = reg_fn(reg_pred[valid].squeeze(-1), reg_targets)
        loss = loss_ce + lambda_reg * loss_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits[valid].argmax(dim=1)
        correct += (preds == nipple_labels[valid]).sum().item()
        total += valid.sum().item()

    return running_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_band=False, sigma=1.0, lambda_reg=0.3):
    model.eval()
    reg_fn = nn.SmoothL1Loss()
    val_loss = 0.0
    correct = 0
    total = 0

    for images, band_images, labels in tqdm(loader, desc="Val"):
        nipple_labels = labels['nipple'].to(device)
        valid = nipple_labels != -1
        if valid.sum() == 0:
            continue

        input_img = band_images.to(device) if use_band else images.to(device)
        logits, reg_pred = model(input_img)

        loss_ce = ordinal_ce_loss(logits[valid], nipple_labels[valid], num_classes=4, sigma=sigma)
        reg_targets = nipple_labels[valid].float() / 3.0
        loss_reg = reg_fn(reg_pred[valid].squeeze(-1), reg_targets)
        val_loss += (loss_ce + lambda_reg * loss_reg).item()

        preds = logits[valid].argmax(dim=1)
        correct += (preds == nipple_labels[valid]).sum().item()
        total += valid.sum().item()

    return val_loss / max(len(loader), 1), correct / max(total, 1)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"nipple_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )

    train_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 始终加载 band 图像，通过 use_band 控制是否用于训练
    train_band = os.path.join(args.band_dir, "train_band") if args.band_dir else None
    val_band = os.path.join(args.band_dir, "val_band") if args.band_dir else None
    use_band = (args.mode == "band")

    train_dataset = ClassificationDataset(
        annotation=args.train_ann, root=args.train_dir,
        transform=train_transform, band_dir=train_band,
    )
    val_dataset = ClassificationDataset(
        annotation=args.val_ann, root=args.val_dir,
        transform=val_transform, band_dir=val_band,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = NippleOnlyModel(num_classes=4, pretrained=True).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_acc = 0.0
    patience_counter = 0
    logging.info(f"Mode: {args.mode} | Band dir: {args.band_dir or 'N/A'} | Use band: {use_band} | Patience: {args.patience}")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device, use_band=use_band, sigma=args.sigma, lambda_reg=args.lambda_reg)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, args.device, use_band=use_band, sigma=args.sigma, lambda_reg=args.lambda_reg)

        logging.info(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.output_dir, f"nipple_{args.mode}_best.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best Val Acc: {best_val_acc:.4f} -> saved {save_path}")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter}/{args.patience} epochs")
            if patience_counter >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    logging.info(f"Done. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="band", choices=["original", "band"],
                        help="original: nipple 用原图; band: nipple 用 band 裁剪")
    parser.add_argument("--train_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/train_classification.json")
    parser.add_argument("--train_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/train")
    parser.add_argument("--val_ann", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json")
    parser.add_argument("--val_dir", type=str, default="/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val")
    parser.add_argument("--band_dir", type=str, default="/home/suzhiling/efficientnet/bands_v2", help="band 裁剪根目录，下含 train/ val/")
    parser.add_argument("--output_dir", type=str, default="./work_dir/models/nipple_test_band/V6")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值，连续无改善则停止")
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian sigma for ordinal soft labels")
    parser.add_argument("--lambda_reg", type=float, default=0.3, help="Weight for regression loss")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()
    main(args)
