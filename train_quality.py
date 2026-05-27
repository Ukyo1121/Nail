"""
单独训练 quality 属性的分类模型。
模型架构与 train_classification.py 一致（MobileNetV3-Large），仅保留一个分类头（2类）。
"""

import os
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from shape_tubes_dataset import build_classification_datasets


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


def make_ordinal_soft_targets(labels, num_classes, sigma=1.0, label_smoothing=0.0):
    device = labels.device
    mask = labels != -1
    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    classes = torch.arange(num_classes, device=device).float()
    safe_labels_float = safe_labels.float().unsqueeze(1)
    dist_sq = (classes.unsqueeze(0) - safe_labels_float) ** 2
    soft_targets = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
    if label_smoothing > 0:
        soft_targets = (1 - label_smoothing) * soft_targets + label_smoothing / num_classes
    return soft_targets, mask


def ordinal_ce_loss(logits, labels, num_classes, sigma=1.0, label_smoothing=0.0):
    soft_targets, mask = make_ordinal_soft_targets(labels, num_classes, sigma, label_smoothing)
    log_probs = F.log_softmax(logits, dim=1)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)
    if mask.sum() == 0:
        return per_sample_loss.sum() * 0.0
    return per_sample_loss[mask].mean()


def loss_function(quality_logits, quality_reg, quality_labels, lambda_reg=0.1, sigma=0.5, label_smoothing=0.0):
    quality_loss = ordinal_ce_loss(quality_logits, quality_labels, num_classes=2, sigma=sigma, label_smoothing=label_smoothing)

    reg_criterion = nn.SmoothL1Loss()
    mask = quality_labels != -1
    if mask.sum() == 0:
        quality_reg_loss = quality_reg.sum() * 0.0
    else:
        targets = quality_labels[mask].float() / 1.0  # 2类归一化到 [0, 1]
        quality_reg_loss = reg_criterion(quality_reg[mask].squeeze(-1), targets)

    return quality_loss + lambda_reg * quality_reg_loss


def calculate_correct(logits, labels):
    _, predicted_classes = torch.max(logits, 1)
    mask = labels != -1
    correct = (predicted_classes[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


class QualityModel(nn.Module):
    """单任务模型：MobileNetV3-Large backbone + quality 分类头。"""
    def __init__(self, num_quality_classes=2, pretrained=True, freeze_blocks=0, dropout=0.0):
        super().__init__()
        self.backbone = mobilenet_v3_large(pretrained=pretrained)
        in_features = self.backbone.classifier[0].in_features  # 960
        self.backbone.classifier = nn.Identity()

        for i in range(freeze_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.quality_fc = nn.Linear(in_features, num_quality_classes)
        self.quality_reg = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.quality_fc(feat), self.quality_reg(feat)


def train_epoch(model, dataloader, loss_criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for _, (images, labels) in progress_bar:
        images = images.to(device)
        quality_labels = labels['quality'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            quality_logits, quality_reg = model(images)
            loss = loss_criterion(quality_logits, quality_reg, quality_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        c, n = calculate_correct(quality_logits, quality_labels)
        correct += c
        total += n

        acc = correct / max(total, 1)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}'})

    avg_loss = running_loss / len(dataloader)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate_epoch(model, dataloader, loss_criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _, (val_images, val_labels) in progress_bar:
            val_images = val_images.to(device)
            val_quality_labels = val_labels['quality'].to(device)

            val_quality_logits, val_quality_reg = model(val_images)
            val_loss_batch = loss_criterion(val_quality_logits, val_quality_reg, val_quality_labels)
            val_loss += val_loss_batch.item()

            c, n = calculate_correct(val_quality_logits, val_quality_labels)
            correct += c
            total += n

            _, preds = torch.max(val_quality_logits, 1)
            mask = val_quality_labels != -1
            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(val_quality_labels[mask].cpu().numpy())

            progress_bar.set_postfix({
                'loss': f'{val_loss_batch.item():.4f}',
                'acc': f'{correct / max(total, 1):.2f}',
            })

    avg_val_loss = val_loss / len(dataloader)
    acc = correct / max(total, 1)

    preds_all = np.concatenate(all_preds) if all_preds else np.array([])
    labels_all = np.concatenate(all_labels) if all_labels else np.array([])
    if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
        macro_f1 = f1_score(labels_all, preds_all, average='macro')
        qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
    else:
        macro_f1 = 0.0
        qwk = 0.0

    metrics = {'acc': acc, 'f1': macro_f1, 'qwk': qwk}
    return avg_val_loss, metrics


def plot_curves(train_losses, train_accs, val_epochs, val_losses, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5)
    ax1.plot(val_epochs, val_losses, 'o-', label='Val Loss', color='#ff7f0e', linewidth=1.5, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve (Quality)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train Acc', color='#1f77b4', linewidth=1.5)
    ax2.plot(val_epochs, val_accs, 'o-', label='Val Acc', color='#ff7f0e', linewidth=1.5, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve (Quality)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "training_curves_quality.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Training curves saved to {save_path}")


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

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    model = QualityModel(num_quality_classes=2, pretrained=True,
                         freeze_blocks=args.freeze_blocks, dropout=args.dropout)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler()

    def criterion(*loss_args):
        return loss_function(*loss_args, lambda_reg=args.lambda_reg, sigma=args.sigma, label_smoothing=args.label_smoothing)

    best_val_accuracy = 0.0
    early_stopping = EarlyStopping(patience=args.patience)

    train_losses, train_accs = [], []
    val_epochs, val_losses, val_accs = [], [], []

    logging.info(f"Anti-overfitting: freeze_blocks={args.freeze_blocks} | dropout={args.dropout} | label_smoothing={args.label_smoothing}")
    logging.info("Starting Training (Quality only)...")
    for epoch in range(args.epochs):
        avg_train_loss, avg_train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, scaler)
        scheduler.step()
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        logging.info(f"Epoch [{epoch + 1}/{args.epochs}] - Training Loss: {avg_train_loss:.4f} | Training Acc: {avg_train_acc:.4f}")

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            avg_val_loss, metrics = evaluate_epoch(model, val_loader, criterion, args.device)
            val_epochs.append(epoch + 1)
            val_losses.append(avg_val_loss)
            val_accs.append(metrics['acc'])
            logging.info(
                f"Epoch [{epoch + 1}/{args.epochs}] - Validation Loss: {avg_val_loss:.4f} | "
                f"Quality: Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | QWK: {metrics['qwk']:.4f}"
            )

            if metrics['acc'] > best_val_accuracy:
                best_val_accuracy = metrics['acc']
                best_model_path = os.path.join(args.output_dir, f'{args.model_save_name}_best.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Improved validation accuracy! Best model saved to: {best_model_path}")

            if early_stopping.step(metrics['acc']):
                logging.info(f"Early stopping at epoch {epoch + 1}, best Val Acc: {early_stopping.best_score:.4f}")
                break

        last_model_path = os.path.join(args.output_dir, f'{args.model_save_name}_last.pth')
        torch.save(model.state_dict(), last_model_path)
        logging.info(f"Saved model of epoch {epoch + 1} to {last_model_path}")

    plot_curves(train_losses, train_accs, val_epochs, val_losses, val_accs, args.output_dir)
    logging.info("Finished Training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-task quality classification training")

    parser.add_argument('--train_ann', type=str, default='/home/suzhiling/efficientnet/annotations/train_classification_B.json')
    parser.add_argument('--train_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/train')
    parser.add_argument('--old_train_ann', type=str, default='/home/suzhiling/efficientnet/annotations/train_classification_A.json')
    parser.add_argument('--old_train_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/train')
    parser.add_argument('--val_ann', type=str, default='/home/suzhiling/efficientnet/annotations/val_classification_B.json')
    parser.add_argument('--val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/val')
    parser.add_argument('--old_val_ann', type=str, default='/home/suzhiling/efficientnet/annotations/val_classification_A.json')
    parser.add_argument('--old_val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val')
    parser.add_argument('--output_dir', type=str, default='./work_dir/models/quality/V1/')
    parser.add_argument('--model_save_name', type=str, default='quality_model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval_freq', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--lambda_reg', type=float, default=0.3)
    parser.add_argument('--freeze_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
