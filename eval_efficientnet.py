import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, cohen_kappa_score

from shape_tubes_dataset import ClassificationDataset
from train_efficientnet import MultiTaskEfficientNetB0, loss_function, calculate_correct


def evaluate(model, dataloader, device):
    model.eval()

    task_names = ['venous', 'nipple', 'arrangement', 'base_transparency']
    task_correct = {t: 0 for t in task_names}
    task_total = {t: 0 for t in task_names}
    task_tp = {t: {} for t in task_names}
    task_fn = {t: {} for t in task_names}
    all_preds = {t: [] for t in task_names}
    all_labels = {t: [] for t in task_names}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            venous_labels = labels['venous'].to(device)
            nipple_labels = labels['nipple'].to(device)
            arrangement_labels = labels['arrangement'].to(device)
            base_transparency_labels = labels['base_transparency'].to(device)

            venous_logits, nipple_logits, arrangement_logits, base_transparency_logits, _, _, _, _ = model(images)

            for logits, label_t, name in [
                (venous_logits, venous_labels, 'venous'),
                (nipple_logits, nipple_labels, 'nipple'),
                (arrangement_logits, arrangement_labels, 'arrangement'),
                (base_transparency_logits, base_transparency_labels, 'base_transparency'),
            ]:
                _, preds = torch.max(logits, 1)
                mask = label_t != -1
                task_correct[name] += (preds[mask] == label_t[mask]).sum().item()
                task_total[name] += mask.sum().item()

                all_preds[name].append(preds[mask].cpu().numpy())
                all_labels[name].append(label_t[mask].cpu().numpy())

                for c in label_t[mask].unique().tolist():
                    c = int(c)
                    gt_c = label_t[mask] == c
                    pred_c = preds[mask] == c
                    tp = (gt_c & pred_c).sum().item()
                    fn = (gt_c & ~pred_c).sum().item()
                    task_tp[name][c] = task_tp[name].get(c, 0) + tp
                    task_fn[name][c] = task_fn[name].get(c, 0) + fn

    print()
    for name in task_names:
        acc = task_correct[name] / task_total[name] if task_total[name] > 0 else 0.0
        all_classes = sorted(set(task_tp[name].keys()) | set(task_fn[name].keys()))
        class_recalls = []
        recall_strs = []
        for c in all_classes:
            tp = task_tp[name].get(c, 0)
            fn = task_fn[name].get(c, 0)
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_recalls.append(r)
            recall_strs.append(f"C{c}={r:.4f}")
        macro_recall = sum(class_recalls) / len(class_recalls) if class_recalls else 0.0

        preds_all = np.concatenate(all_preds[name]) if all_preds[name] else np.array([])
        labels_all = np.concatenate(all_labels[name]) if all_labels[name] else np.array([])
        if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
            macro_f1 = f1_score(labels_all, preds_all, average='macro')
            qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
        else:
            macro_f1 = 0.0
            qwk = 0.0

        print(f"  {name:>20s}  Acc: {acc:.4f} | F1: {macro_f1:.4f} | QWK: {qwk:.4f}  ({task_correct[name]}/{task_total[name]})  Recall(macro): {macro_recall:.4f}  [{', '.join(recall_strs)}]")

    avg_acc = sum(task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0 for t in task_names) / len(task_names)
    avg_recall = 0.0
    avg_f1 = 0.0
    avg_qwk = 0.0
    for t in task_names:
        all_classes = sorted(set(task_tp[t].keys()) | set(task_fn[t].keys()))
        cr = []
        for c in all_classes:
            tp = task_tp[t].get(c, 0)
            fn = task_fn[t].get(c, 0)
            cr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        avg_recall += sum(cr) / len(cr) if cr else 0.0

        preds_all = np.concatenate(all_preds[t]) if all_preds[t] else np.array([])
        labels_all = np.concatenate(all_labels[t]) if all_labels[t] else np.array([])
        if len(labels_all) > 0 and len(np.unique(labels_all)) > 1:
            avg_f1 += f1_score(labels_all, preds_all, average='macro')
            avg_qwk += cohen_kappa_score(labels_all, preds_all, weights='quadratic')

    avg_recall /= len(task_names)
    avg_f1 /= len(task_names)
    avg_qwk /= len(task_names)
    print(f"  {'Average':>20s}  Acc: {avg_acc:.4f} | F1: {avg_f1:.4f} | QWK: {avg_qwk:.4f}  Recall(macro): {avg_recall:.4f}")


def visualize(model, dataset, device, vis_dir, max_samples=50):
    """将每张图的四个属性的GT和预测结果绘制在原图上并保存。"""
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()

    task_names = ['venous', 'nipple', 'arrangement', 'base_transparency']

    eval_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    count = 0
    for idx in tqdm(range(len(dataset)), desc="Visualizing"):
        if count >= max_samples:
            break

        # 从 dataset 的 coco 中获取图像路径
        image_id = dataset.image_ids[idx]
        img_info = dataset.coco.loadImgs(image_id)[0]
        img_path = os.path.join(dataset.root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取 GT 标签
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        annotations = dataset.coco.loadAnns(ann_ids)
        gt = {'venous': -1, 'nipple': -1, 'arrangement': -1, 'base_transparency': -1}
        for ann in annotations:
            attr_name = dataset.cat_id_to_attr.get(ann['category_id'])
            if attr_name and attr_name in gt:
                gt[attr_name] = int(ann['attributes'][attr_name])

        # 模型推理
        pil_img = Image.fromarray(img_rgb)
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            v_logits, n_logits, a_logits, b_logits, _, _, _, _ = model(input_tensor)
        preds = {}
        for logits, name in [(v_logits, 'venous'), (n_logits, 'nipple'),
                              (a_logits, 'arrangement'), (b_logits, 'base_transparency')]:
            _, pred = torch.max(logits, 1)
            preds[name] = pred.item()

        # 在图上绘制文字
        h, w = img.shape[:2]
        overlay = img.copy()
        bar_h = 30 * len(task_names) + 20
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        img_show = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        for i, name in enumerate(task_names):
            gt_val = gt[name]
            pred_val = preds[name]
            gt_str = str(gt_val) if gt_val != -1 else "N/A"
            correct = gt_val == pred_val
            color = (0, 255, 0) if correct else (0, 0, 255)  # 绿色=正确 红色=错误
            if gt_val == -1:
                color = (200, 200, 200)  # 灰色=无GT

            text = f"{name}: GT={gt_str}  Pred={pred_val}"
            y = 30 * (i + 1)
            cv2.putText(img_show, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)

        save_name = img_info['file_name']
        save_path = os.path.join(vis_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_show)
        count += 1

    print(f"\nVisualization saved to {vis_dir} ({count} images)")


def main(args):
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = ClassificationDataset(
        annotation=args.val_ann,
        root=args.val_dir,
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = MultiTaskEfficientNetB0(4, 4, 4, 3, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    print(f"Loaded model from {args.model_path}")
    evaluate(model, val_loader, args.device)

    if args.vis_dir:
        visualize(model, val_dataset, args.device, args.vis_dir, max_samples=args.vis_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate multi-task EfficientNet-B0")
    parser.add_argument('--model_path', type=str, default='./work_dir/models/classification/V6/effiecientnet_classification_best.pth', help="Path to trained .pth model")
    parser.add_argument('--val_ann', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/annotations/val_classification.json')
    parser.add_argument('--val_dir', type=str, default='/data/zhangxiaohao/dazhouV2/Aclass/all_new/output/val')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--vis_dir', type=str, default= None, help="Directory to save visualization results")
    parser.add_argument('--vis_max', type=int, default=500, help="Max number of images to visualize")

    args = parser.parse_args()
    main(args)
