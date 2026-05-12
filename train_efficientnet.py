import os

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
            return False  # 不停止
        self.counter += 1
        return self.counter >= self.patience  # True = 应该停止


def make_ordinal_soft_targets(labels, num_classes, sigma=1.0):
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
    return soft_targets, mask


def ordinal_ce_loss(logits, labels, num_classes, sigma=1.0):
    """Ordinal-aware cross entropy: 用高斯软标签替代 one-hot，让邻近类的惩罚更小。"""
    soft_targets, mask = make_ordinal_soft_targets(labels, num_classes, sigma)
    log_probs = F.log_softmax(logits, dim=1)  # (B, C)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)  # (B,)
    if mask.sum() == 0:
        return per_sample_loss.sum() * 0.0
    return per_sample_loss[mask].mean()


def loss_function(venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
                  venous_reg, nipple_reg, arrangement_reg, base_transparency_reg,
                  venous_labels, nipple_labels, arrangement_labels, base_transparency_labels,
                  lambda_reg=0.1, sigma=0.5):
    venous_loss = ordinal_ce_loss(venous_logits, venous_labels, num_classes=4, sigma=sigma)
    nipple_loss = ordinal_ce_loss(nipple_logits, nipple_labels, num_classes=4, sigma=sigma)
    arrangement_loss = ordinal_ce_loss(arrangement_logits, arrangement_labels, num_classes=4, sigma=sigma)
    base_transparency_loss = ordinal_ce_loss(base_transparency_logits, base_transparency_labels, num_classes=3, sigma=sigma)

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


def calculate_accuracy(logits, labels):
    _, predicted_classes = torch.max(logits, 1)
    mask = labels != -1
    if mask.sum().item() == 0:
        return 0.0
    correct_predictions = (predicted_classes[mask] == labels[mask]).sum().item()
    total_samples = mask.sum().item()
    accuracy = correct_predictions / total_samples
    return accuracy


class MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, num_venous_classes, num_nipple_classes, num_arrangement_classes, num_base_transparency_classes, pretrained=True):
        super(MultiTaskEfficientNetB0, self).__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

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
        return (self.venous_fc(feat), self.nipple_fc(feat),
                self.arrangement_fc(feat), self.base_transparency_fc(feat),
                self.venous_reg(feat), self.nipple_reg(feat),
                self.arrangement_reg(feat), self.base_transparency_reg(feat))


def train_epoch(model, dataloader, loss_criterion, optimizer, device):
    """
    训练模型一个 epoch.

    Args:
        model: 待训练的模型.
        dataloader: 训练数据 DataLoader.
        loss_criterion: 损失函数.
        optimizer: 优化器.
        device: 设备 (CPU 或 CUDA).
        schedular: 学习率调度器 (可选).

    Returns:
        avg_loss: 本 epoch 的平均训练损失.
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training")
    for _, (images, labels) in progress_bar:
        images = images.to(device)
        venous_labels = labels['venous'].to(device)
        nipple_labels = labels['nipple'].to(device)
        arrangement_labels = labels['arrangement'].to(device)
        base_transparency_labels = labels['base_transparency'].to(device)

        optimizer.zero_grad()

        (venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
         venous_reg, nipple_reg, arrangement_reg, base_transparency_reg) = model(images)
        loss = loss_criterion(venous_logits, nipple_logits, arrangement_logits, base_transparency_logits,
                              venous_reg, nipple_reg, arrangement_reg, base_transparency_reg,
                              venous_labels, nipple_labels, arrangement_labels, base_transparency_labels)

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

        # 更新 tqdm 进度条的 postfix (显示loss信息)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # 显示当前 batch loss

    avg_loss = running_loss / len(dataloader)  # 计算平均 epoch loss
    return avg_loss


def evaluate_epoch(model, dataloader, loss_criterion, device, calculate_accuracy_fn):
    """
    评估模型在一个 epoch 的性能 (在验证集或测试集上).

    Args:
        model: 待评估的模型.
        dataloader: 验证/测试数据 DataLoader.
        loss_criterion: 损失函数.
        device: 设备 (CPU 或 CUDA).
        calculate_accuracy_fn: 计算准确率的函数.

    Returns:
        avg_loss: 本 epoch 的平均验证/测试损失.
        arrangement_accuracy: Arrangement 分类准确率.
        nipple_accuracy: Nipple 分类准确率.
    """
    model.eval()
    val_loss = 0.0
    venous_accuracies = []
    nipple_accuracies = []
    arrangement_accuracies = []
    base_transparency_accuracies = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation")
    with torch.no_grad():
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

            venous_acc = calculate_accuracy_fn(val_venous_logits, val_venous_labels)
            nipple_acc = calculate_accuracy_fn(val_nipple_logits, val_nipple_labels)
            arrangement_acc = calculate_accuracy_fn(val_arrangement_logits, val_arrangement_labels)
            base_transparency_acc = calculate_accuracy_fn(val_base_transparency_logits, val_base_transparency_labels)

            venous_accuracies.append(venous_acc)
            nipple_accuracies.append(nipple_acc)
            arrangement_accuracies.append(arrangement_acc)
            base_transparency_accuracies.append(base_transparency_acc)

            progress_bar.set_postfix({
                'loss': f'{val_loss_batch.item():.4f}',
                'ven': f'{venous_acc:.2f}',
                'nip': f'{nipple_acc:.2f}',
                'arr': f'{arrangement_acc:.2f}',
                'btr': f'{base_transparency_acc:.2f}'
            })

    avg_val_loss = val_loss / len(dataloader)
    venous_val_accuracy = sum(venous_accuracies) / len(venous_accuracies)
    nipple_val_accuracy = sum(nipple_accuracies) / len(nipple_accuracies)
    arrangement_val_accuracy = sum(arrangement_accuracies) / len(arrangement_accuracies)
    base_transparency_val_accuracy = sum(base_transparency_accuracies) / len(base_transparency_accuracies)

    return avg_val_loss, venous_val_accuracy, nipple_val_accuracy, arrangement_val_accuracy, base_transparency_val_accuracy


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
        transforms.Resize((1024, 1024)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset ---
    train_dataset = ClassificationDataset(
        annotation=args.train_ann,
        root=args.train_dir,
        transform=train_transform
    )
    val_dataset = ClassificationDataset(
        annotation=args.val_ann,
        root=args.val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # --- 模型初始化 ---
    model = MultiTaskEfficientNetB0(4, 4, 4, 3, pretrained=True)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    def criterion(*loss_args):
        return loss_function(*loss_args, lambda_reg=args.lambda_reg, sigma=args.sigma)

    best_val_accuracy = 0.0
    early_stopping = EarlyStopping(patience=args.patience)

    logging.info("Starting Training...")
    for epoch in range(args.epochs):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        scheduler.step()
        logging.info(f"Epoch [{epoch + 1}/{args.epochs}] - Training Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            avg_val_loss, venous_val_acc, nipple_val_acc, arrangement_val_acc, base_transparency_val_acc = evaluate_epoch(
                model, val_loader, criterion, args.device, calculate_accuracy
            )
            total_val_acc = (venous_val_acc + nipple_val_acc + arrangement_val_acc + base_transparency_val_acc) / 4
            logging.info(
                f"Epoch [{epoch + 1}/{args.epochs}] - Validation Loss: {avg_val_loss:.4f} | "
                f"Venous Acc: {venous_val_acc:.4f} | "
                f"Nipple Acc: {nipple_val_acc:.4f} | "
                f"Arrangement Acc: {arrangement_val_acc:.4f} | "
                f"BaseTransparency Acc: {base_transparency_val_acc:.4f} | "
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

    parser.add_argument('--train_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/train_classification.json', help="Path to training annotation file")
    parser.add_argument('--train_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/train', help="Path to training image directory")
    parser.add_argument('--val_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json', help="Path to validation annotation file")
    parser.add_argument('--val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val', help="Path to validation image directory")
    parser.add_argument('--output_dir', type=str, default='./work_dir/models/classification/V7/', help="Directory to save models and logs")
    parser.add_argument('--model_save_name', type=str, default='effiecientnet_classification',
                        help="Directory to save models and logs")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval_freq', type=int, default=4, help="Evaluate every N epochs")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience (eval cycles)")
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian sigma for ordinal soft labels")
    parser.add_argument('--lambda_reg', type=float, default=0.3, help="Weight for regression loss")

    args = parser.parse_args()
    main(args)
