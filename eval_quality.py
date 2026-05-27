"""
单独评估 quality 属性的分类模型。
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, cohen_kappa_score
from datetime import datetime
from shape_tubes_dataset import build_classification_datasets
from train_quality import QualityModel


def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    offset_correct = 0
    total = 0
    tp = {}
    fn = {}
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            quality_labels = labels['quality'].to(device)

            quality_logits, _ = model(images)

            _, preds = torch.max(quality_logits, 1)
            mask = quality_labels != -1
            correct += (preds[mask] == quality_labels[mask]).sum().item()
            offset_correct += ((preds[mask] - quality_labels[mask]).abs() <= 1).sum().item()
            total += mask.sum().item()

            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(quality_labels[mask].cpu().numpy())

            for c in quality_labels[mask].unique().tolist():
                c = int(c)
                gt_c = quality_labels[mask] == c
                pred_c = preds[mask] == c
                tp[c] = tp.get(c, 0) + (gt_c & pred_c).sum().item()
                fn[c] = fn.get(c, 0) + (gt_c & ~pred_c).sum().item()

    acc = correct / total if total > 0 else 0.0
    all_classes = sorted(set(tp.keys()) | set(fn.keys()))
    class_recalls = []
    recall_strs = []
    for c in all_classes:
        tp_c = tp.get(c, 0)
        fn_c = fn.get(c, 0)
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        class_recalls.append(r)
        recall_strs.append(f"C{c}={r:.4f}")
    macro_recall = sum(class_recalls) / len(class_recalls) if class_recalls else 0.0

    preds_all = np.concatenate(all_preds) if all_preds else np.array([])
    labels_all = np.concatenate(all_labels) if all_labels else np.array([])
    if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
        macro_f1 = f1_score(labels_all, preds_all, average='macro')
        qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
    else:
        macro_f1 = 0.0
        qwk = 0.0

    offset_acc = offset_correct / total if total > 0 else 0.0
    print(f"\n  {'quality':>20s}  Acc: {acc:.4f} | Off1: {offset_acc:.4f} | F1: {macro_f1:.4f} | QWK: {qwk:.4f}  ({correct}/{total})  Recall(macro): {macro_recall:.4f}  [{', '.join(recall_strs)}]")


def _resolve_dataset(dataset, idx):
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        offset = 0
        for ds in dataset.datasets:
            if idx < offset + len(ds):
                return ds, idx - offset
            offset += len(ds)
        raise IndexError(f"Index {idx} out of range")
    return dataset, idx


def visualize(model, dataset, device, vis_dir, max_samples=50):
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    count = 0
    for idx in tqdm(range(len(dataset)), desc="Visualizing"):
        if count >= max_samples:
            break

        ds, inner_idx = _resolve_dataset(dataset, idx)

        image_id = ds.image_ids[inner_idx]
        img_info = ds.coco.loadImgs(image_id)[0]
        img_path = os.path.join(ds.root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_quality = img_info.get('quality', -1)

        pil_img = Image.fromarray(img_rgb)
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            q_logits, _ = model(input_tensor)
        _, pred = torch.max(q_logits, 1)
        pred_quality = pred.item()

        h, w = img.shape[:2]
        overlay = img.copy()
        bar_h = 50
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        img_show = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        gt_str = str(gt_quality) if gt_quality != -1 else "N/A"
        correct = gt_quality == pred_quality
        if gt_quality == -1:
            color = (200, 200, 200)
        elif correct:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        text = f"Quality: GT={gt_str}  Pred={pred_quality}"
        cv2.putText(img_show, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

        save_name = img_info['file_name']
        save_path = os.path.join(vis_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_show)
        count += 1

    print(f"\nVisualization saved to {vis_dir} ({count} images)")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main(args):
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset, _ = build_classification_datasets(
        train_ann=args.val_ann,
        train_dir=args.val_dir,
        transform=val_transform,
        old_train_ann=args.old_val_ann,
        old_train_dir=args.old_val_dir,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = QualityModel(num_quality_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    if args.log_file is None:
        log_dir = os.path.join(os.path.dirname(args.model_path), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'eval_logs.txt')
    else:
        log_path = args.log_file

    log_f = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_f)

    print(f"\n{'='*60}")
    print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Loaded model from {args.model_path}")
    evaluate(model, val_loader, args.device)

    if args.vis_dir:
        visualize(model, val_dataset, args.device, args.vis_dir, max_samples=args.vis_max)

    sys.stdout = original_stdout
    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate quality classification model")
    parser.add_argument('--model_path', type=str, default='./work_dir/models/quality/V1/quality_model_best.pth')
    parser.add_argument('--val_ann', type=str, default='/home/suzhiling/efficientnet/annotations/val_classification_A.json')
    parser.add_argument('--val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val')
    parser.add_argument('--old_val_ann', type=str, default='/home/suzhiling/efficientnet/annotations/val_classification_B.json')
    parser.add_argument('--old_val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/val')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--vis_dir', type=str, default='vis_quality_v1')
    parser.add_argument('--vis_max', type=int, default=500)
    parser.add_argument('--log_file', type=str, default=None)

    args = parser.parse_args()
    main(args)
