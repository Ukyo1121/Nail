import os

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F  # 导入 functional 模块
from torch import optim
import torch
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
import math

from shape_tubes_dataset import ClassificationDataset, build_classification_datasets


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
            return False  # 不停止
        self.counter += 1
        return self.counter >= self.patience  # True = 应该停止


def make_ordinal_soft_targets(labels, num_classes, sigma=1.0, label_smoothing=0.0):
    """为每个标签生成高斯软标签分布，距离越近权重越大。
    例如标签 0 (4类): [0.607, 0.242, 0.018, 0.000] — 邻近类有梯度，远处几乎为0。
    """
    # labels: (B,) 整数标签，可能包含 -1
    device = labels.device
    mask = labels != -1
    # 用0占位，后面只计算有效样本的loss
    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    classes = torch.arange(num_classes, device=device).float()  # (C,)
    safe_labels_float = safe_labels.float().unsqueeze(1)        # (B, 1)
    dist_sq = (classes.unsqueeze(0) - safe_labels_float) ** 2   # (B, C)
    soft_targets = torch.exp(-dist_sq / (2 * sigma ** 2))       # (B, C)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)  # 归一化
    if label_smoothing > 0:
        soft_targets = (1 - label_smoothing) * soft_targets + label_smoothing / num_classes
    return soft_targets, mask


def ordinal_ce_loss(logits, labels, num_classes, sigma=1.0, label_smoothing=0.0):
    """Ordinal-aware cross entropy: 用高斯软标签替代 one-hot，让邻近类的惩罚更小。"""
    soft_targets, mask = make_ordinal_soft_targets(labels, num_classes, sigma, label_smoothing)
    log_probs = F.log_softmax(logits, dim=1)  # (B, C)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)  # (B,)
    if mask.sum() == 0:
        return per_sample_loss.sum() * 0.0
    return per_sample_loss[mask].mean()


def loss_function(venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
                  venous_reg, nipple_reg, arrangement_reg, base_transparency_reg,
                  venous_labels, nipple_labels, arrangement_labels, base_transparency_labels,
                  lambda_reg=0.1, sigma=0.5, label_smoothing=0.0):
    venous_loss = ordinal_ce_loss(venous_logits, venous_labels, num_classes=4, sigma=sigma, label_smoothing=label_smoothing)
    nipple_loss = ordinal_ce_loss(nipple_logits, nipple_labels, num_classes=4, sigma=sigma, label_smoothing=label_smoothing)
    arrangement_loss = ordinal_ce_loss(arrangement_logits, arrangement_labels, num_classes=4, sigma=sigma, label_smoothing=label_smoothing)
    base_transparency_loss = ordinal_ce_loss(base_transparency_logits, base_transparency_labels, num_classes=3, sigma=sigma, label_smoothing=label_smoothing)

    reg_criterion = nn.SmoothL1Loss()

    def masked_reg_loss(reg_out, labels, num_classes):
        mask = labels != -1
        if mask.sum() == 0:
            return reg_out.sum() * 0.0
        targets = labels[mask].float() / (num_classes - 1)
        return reg_criterion(reg_out[mask].squeeze(-1), targets)

    venous_reg_loss = masked_reg_loss(venous_reg, venous_labels, 4)
    nipple_reg_loss = masked_reg_loss(nipple_reg, nipple_labels, 4)
    arrangement_reg_loss = masked_reg_loss(arrangement_reg, arrangement_labels, 4)
    base_transparency_reg_loss = masked_reg_loss(base_transparency_reg, base_transparency_labels, 3)

    return (venous_loss + nipple_loss + arrangement_loss + base_transparency_loss
            + lambda_reg * (venous_reg_loss + nipple_reg_loss + arrangement_reg_loss + base_transparency_reg_loss))


def calculate_correct(logits, labels):
    _, predicted_classes = torch.max(logits, 1)
    mask = labels != -1
    correct = (predicted_classes[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


class MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, num_venous_classes, num_nipple_classes, num_arrangement_classes, num_base_transparency_classes,
                 pretrained=True, freeze_blocks=0, dropout=0.0):
        super(MultiTaskEfficientNetB0, self).__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # 冻结 backbone 前 freeze_blocks 层
        for i in range(freeze_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 分类头
        self.venous_fc = nn.Linear(in_features, num_venous_classes)
        self.nipple_fc = nn.Linear(in_features, num_nipple_classes)
        self.arrangement_fc = nn.Linear(in_features, num_arrangement_classes)
        self.base_transparency_fc = nn.Linear(in_features, num_base_transparency_classes)

        # 回归头
        self.venous_reg = nn.Linear(in_features, 1)
        self.nipple_reg = nn.Linear(in_features, 1)
        self.arrangement_reg = nn.Linear(in_features, 1)
        self.base_transparency_reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return (self.venous_fc(feat), self.nipple_fc(feat),
                self.arrangement_fc(feat), self.base_transparency_fc(feat),
                self.venous_reg(feat), self.nipple_reg(feat),
                self.arrangement_reg(feat), self.base_transparency_reg(feat))


def train_epoch(model, dataloader, loss_criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training")
    for _, (images, labels) in progress_bar:
        images = images.to(device)
        venous_labels = labels['venous'].to(device)
        nipple_labels = labels['nipple'].to(device)
        arrangement_labels = labels['arrangement'].to(device)
        base_transparency_labels = labels['base_transparency'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            (venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
             venous_reg, nipple_reg, arrangement_reg, base_transparency_reg) = model(images)
            loss = loss_criterion(venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
                                  venous_reg, nipple_reg, arrangement_reg, base_transparency_reg,
                                  venous_labels, nipple_labels, arrangement_labels, base_transparency_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate_epoch(model, dataloader, loss_criterion, device):
    model.eval()
    val_loss = 0.0
    correct = {'venous': 0, 'nipple': 0, 'arrangement': 0, 'base_transparency': 0}
    total = {'venous': 0, 'nipple': 0, 'arrangement': 0, 'base_transparency': 0}
    all_preds = {'venous': [], 'nipple': [], 'arrangement': [], 'base_transparency': []}
    all_labels = {'venous': [], 'nipple': [], 'arrangement': [], 'base_transparency': []}

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _, (val_images, val_labels) in progress_bar:
            val_images = val_images.to(device)
            val_venous_labels = val_labels['venous'].to(device)
            val_nipple_labels = val_labels['nipple'].to(device)
            val_arrangement_labels = val_labels['arrangement'].to(device)
            val_base_transparency_labels = val_labels['base_transparency'].to(device)

            (val_venous_logits, val_nipple_logits, val_arrangement_logits, val_base_transparency_logits,
             val_venous_reg, val_nipple_reg, val_arrangement_reg, val_base_transparency_reg) = model(val_images)
            val_loss_batch = loss_criterion(val_venous_logits, val_nipple_logits, val_arrangement_logits, val_base_transparency_logits,
                                            val_venous_reg, val_nipple_reg, val_arrangement_reg, val_base_transparency_reg,
                                            val_venous_labels, val_nipple_labels, val_arrangement_labels, val_base_transparency_labels)
            val_loss += val_loss_batch.item()

            for name, logits, labels_t in [
                ('venous', val_venous_logits, val_venous_labels),
                ('nipple', val_nipple_logits, val_nipple_labels),
                ('arrangement', val_arrangement_logits, val_arrangement_labels),
                ('base_transparency', val_base_transparency_logits, val_base_transparency_labels),
            ]:
                c, n = calculate_correct(logits, labels_t)
                correct[name] += c
                total[name] += n

                _, preds = torch.max(logits, 1)
                mask = labels_t != -1
                all_preds[name].append(preds[mask].cpu().numpy())
                all_labels[name].append(labels_t[mask].cpu().numpy())

            progress_bar.set_postfix({
                'loss': f'{val_loss_batch.item():.4f}',
                'ven': f'{correct["venous"] / max(total["venous"], 1):.2f}',
                'nip': f'{correct["nipple"] / max(total["nipple"], 1):.2f}',
                'arr': f'{correct["arrangement"] / max(total["arrangement"], 1):.2f}',
                'btr': f'{correct["base_transparency"] / max(total["base_transparency"], 1):.2f}',
            })

    avg_val_loss = val_loss / len(dataloader)

    metrics = {}
    for name in ['venous', 'nipple', 'arrangement', 'base_transparency']:
        acc = correct[name] / max(total[name], 1)
        preds_all = np.concatenate(all_preds[name]) if all_preds[name] else np.array([])
        labels_all = np.concatenate(all_labels[name]) if all_labels[name] else np.array([])
        if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
            macro_f1 = f1_score(labels_all, preds_all, average='macro')
            qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
        else:
            macro_f1 = 0.0
            qwk = 0.0
        metrics[name] = {'acc': acc, 'f1': macro_f1, 'qwk': qwk}

    return avg_val_loss, metrics


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # --- 日志设置 ---
    log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    # --- Transform ---
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset ---
    train_dataset, val_dataset = build_classification_datasets(
        train_ann=args.train_ann,
        train_dir=args.train_dir,
        transform=train_transform,
        old_train_ann=args.old_train_ann,
        old_train_dir=args.old_train_dir,
        val_ann=args.val_ann,
        val_dir=args.val_dir,
        old_val_ann=args.old_val_ann,
        old_val_dir=args.old_val_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # --- 模型初始化 ---
    model = MultiTaskEfficientNetB0(4, 4, 4, 3, pretrained=True,
                                    freeze_blocks=args.freeze_blocks, dropout=args.dropout)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler()
    def criterion(*loss_args):
        return loss_function(*loss_args, lambda_reg=args.lambda_reg, sigma=args.sigma, label_smoothing=args.label_smoothing)

    best_val_accuracy = 0.0
    early_stopping = EarlyStopping(patience=args.patience)

    logging.info(f"Anti-overfitting: freeze_blocks={args.freeze_blocks} | dropout={args.dropout} | label_smoothing={args.label_smoothing}")
    logging.info("Starting Training...")
    for epoch in range(args.epochs):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device, scaler)
        scheduler.step()
        logging.info(f"Epoch [{epoch + 1}/{args.epochs}] - Training Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            avg_val_loss, metrics = evaluate_epoch(
                model, val_loader, criterion, args.device
            )
            total_val_acc = sum(m['acc'] for m in metrics.values()) / len(metrics)
            logging.info(
                f"Epoch [{epoch + 1}/{args.epochs}] - Validation Loss: {avg_val_loss:.4f} | "
                f"Venous: Acc: {metrics['venous']['acc']:.4f} | F1: {metrics['venous']['f1']:.4f} | QWK: {metrics['venous']['qwk']:.4f} | "
                f"Nipple: Acc: {metrics['nipple']['acc']:.4f} | F1: {metrics['nipple']['f1']:.4f} | QWK: {metrics['nipple']['qwk']:.4f} | "
                f"Arrangement: Acc: {metrics['arrangement']['acc']:.4f} | F1: {metrics['arrangement']['f1']:.4f} | QWK: {metrics['arrangement']['qwk']:.4f} | "
                f"BaseTransparency: Acc: {metrics['base_transparency']['acc']:.4f} | F1: {metrics['base_transparency']['f1']:.4f} | QWK: {metrics['base_transparency']['qwk']:.4f} | "
                f"Avg. Val Acc: {total_val_acc:.4f}"
            )

            if total_val_acc > best_val_accuracy:
                best_val_accuracy = total_val_acc
                best_model_path = os.path.join(args.output_dir, f'{args.model_save_name}_best.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Improved validation accuracy! Best model saved to: {best_model_path}")

            if early_stopping.step(total_val_acc):
                logging.info(f"Early stopping at epoch {epoch + 1}, best Val Acc: {early_stopping.best_score:.4f}")
                break

        last_model_path = os.path.join(args.output_dir, f'{args.model_save_name}_last.pth')
        torch.save(model.state_dict(), last_model_path)
        logging.info(f"Saved model of epoch {epoch + 1} to {last_model_path}")

    logging.info("Finished Training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-task classification training with EfficientNet-B0")

    parser.add_argument('--train_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/annotations/train_classification.json', help="Path to training annotation file")
    parser.add_argument('--train_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/train', help="Path to training image directory")
    parser.add_argument('--old_train_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/train_classification.json', help="Path to old training annotation file (optional, for concatenation)")
    parser.add_argument('--old_train_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/train', help="Path to old training image directory (optional, for concatenation)")
    parser.add_argument('--val_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/annotations/val_classification.json', help="Path to validation annotation file")
    parser.add_argument('--val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/val', help="Path to validation image directory")
    parser.add_argument('--old_val_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json', help="Path to old validation annotation file (optional, for concatenation)")
    parser.add_argument('--old_val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val', help="Path to old validation image directory (optional, for concatenation)")
    parser.add_argument('--output_dir', type=str, default='./work_dir/models/classification/V8/', help="Directory to save models and logs")
    parser.add_argument('--model_save_name', type=str, default='effiecientnet_classification',
                        help="Directory to save models and logs")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval_freq', type=int, default=4, help="Evaluate every N epochs")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience (eval cycles)")
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian sigma for ordinal soft labels")
    parser.add_argument('--lambda_reg', type=float, default=0.3, help="Weight for regression loss")
    parser.add_argument('--freeze_blocks', type=int, default=2, help="冻结 backbone 前 N 层 (0 表示不冻结)")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate (0 表示不添加)")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="标签平滑系数 (0 表示不平滑)")

    args = parser.parse_args()
    main(args)
