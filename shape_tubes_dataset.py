import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from PIL import Image
import json

from torchvision import transforms

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:
    _HAS_ALB = False
# from utils import detection_collate_fn


class ShapeTubesDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # 只保留我们需要的类别
        self.class_ids = self.coco.getCatIds(catNms=['shape_tubes', 'shape_tubes_unclear'])

    def __getitem__(self, idx):
        # 获取图像ID
        img_id = self.ids[idx]

        # 获取图像信息
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)
        anns = self.coco.loadAnns(ann_ids)

        # 初始化目标字典
        target = {
            'boxes': [],
            'labels': [],
            'masks': [],
            'attributes': {
                'cross_tube': [],
                'malformed_tube': [],
                'bleed_speed': [],
                'rbc_aggregation': []
            },
            'image_id': torch.tensor([img_id])
        }

        for ann in anns:
            # 类别处理 (1 for shape_tubes, 2 for shape_tubes_unclear)
            label = 1 if ann['category_id'] == self.coco.getCatIds(catNms=['shape_tubes'])[0] else 2
            target['labels'].append(label)

            # 处理bbox - 对于shape_tubes需要从多边形生成bbox
            if 'bbox' in ann and len(ann['bbox']) == 4:
                bbox = ann['bbox']
            else:
                # 从多边形生成bbox
                mask = self.coco.annToMask(ann)
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            target['boxes'].append(bbox)

            # 处理mask - 只有shape_tubes需要
            if label == 1:
                mask = self.coco.annToMask(ann)
                target['masks'].append(mask)
            else:
                # 对于shape_tubes_unclear，添加一个空的mask
                target['masks'].append(np.zeros((img_info['height'], img_info['width']), dtype=np.uint8))

            # 处理属性
            attributes = ann.get('attributes', {})
            target['attributes']['cross_tube'].append(int(attributes.get('cross_tube', 0)))
            target['attributes']['malformed_tube'].append(int(attributes.get('malformed_tube', 0)))
            target['attributes']['bleed_speed'].append(int(attributes.get('bleed_speed', 0)))
            target['attributes']['rbc_aggregation'].append(int(attributes.get('rbc_aggregation', 0)))

        # 转换为tensor
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['masks'] = torch.as_tensor(np.stack(target['masks']), dtype=torch.uint8)

        # 转换属性
        for k in target['attributes']:
            target['attributes'][k] = torch.as_tensor(target['attributes'][k], dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def transform_detection(train=True):
    transforms_list = []

    if train:
        transforms_list.extend([
            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),

            # 颜色增强 - 这个是安全的，不影响边界框
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),

            # 随机调整图像的锐度 - 这个也是安全的
            transforms.RandomAdjustSharpness(
                sharpness_factor=2,
                p=0.5
            ),

            # 随机高斯模糊
            transforms.GaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0)
            ),
        ])

    # 测试时只进行resize
    transforms_list.extend([
        transforms.Resize(size=(640, 640), antialias=True),
        transforms.ToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transforms_list)


class DetectionDataset(Dataset):
    def __init__(self, annotation, root, img_size=640, transform=None, target_classes=None, train=True):
        """
        自定义 COCO 数据集，过滤掉无关类别
        :param annotation: COCO 标注文件路径
        :param root: 图片存放目录
        :param target_classes: 需要检测的类别列表
        :param img_size: 目标图片大小
        :param transform: 预处理变换
        :param train: 是否是训练模式
        """
        if target_classes is None:
            target_classes = ["bws", "exudate", "bleed", "venous"]
        self.img_dir = root
        self.img_size = img_size
        self.train = train

        # 使用新的transform函数
        self.transform = transform if transform is not None else transform_detection(train)

        # 读取 COCO JSON
        with open(annotation, 'r') as f:
            self.coco_data = json.load(f)

        # 获取需要的图片信息
        self.image_infos = self.coco_data["images"]
        # print(len(self.image_infos))

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        h_ori, w_ori = img_info["height"], img_info["width"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # 读取目标框
        annotations = [ann for ann in self.coco_data["annotations"] if ann["image_id"] == img_info["id"]]
        boxes = []
        labels = []
        """for ann in annotations:
            x, y, w, h = ann["bbox"]  # COCO bbox 格式: [x_min, y_min, width, height]
            boxes.append([x, y, x + w, y + h])  # 转换为 YOLO 格式: [x_min, y_min, x_max, y_max]
            labels.append(ann["category_id"])"""
        if len(annotations) > 0:
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
            # 有标注时的处理
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            # 没有标注时的处理
            # print(f'No annotations for image {idx}')
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # 创建空的边界框tensor
            labels = torch.zeros(0, dtype=torch.long)

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h_ori, w_ori))
        targets = {"boxes": boxes, "labels": labels}

        # 预处理
        if self.transform:
            image, targets = self.transform(image, targets)

            # print(boxes)
        # targets["boxes"] = targets["boxes"].data
        return image, targets


class ClassificationDataset(Dataset):
    def __init__(self, annotation, root, transform=None, band_dir=None):
        """
        Args:
            annotation (string): COCO 格式的标注文件路径 (JSON).
            root (string): 图像文件夹路径.
            transform (callable, optional): 应用于图像的 transforms.
            band_dir (string, optional): band 裁剪图像目录，设了则返回 (image, band_image, targets).
        """
        self.annotation = annotation
        self.root = root
        self.band_dir = band_dir
        self.coco = COCO(self.annotation)
        self.image_ids = list(self.coco.imgs.keys())
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        if band_dir is not None:
            self.band_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        with open(self.annotation, 'r') as f:
            self.coco_data = json.load(f)
        self.image_infos = list(self.coco_data['images'])

        self.cat_id_to_attr = {}
        for cat in self.coco_data['categories']:
            self.cat_id_to_attr[cat['id']] = cat['name']

    @staticmethod
    def _apply_transform(transform, image):
        if _HAS_ALB and isinstance(transform, A.Compose):
            image_np = np.array(image)
            result = transform(image=image_np)
            return result['image']
        else:
            return transform(image)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.root, image_filename)
        image = Image.open(image_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        targets = {'venous': -1, 'nipple': -1, 'arrangement': -1, 'base_transparency': -1}
        for ann in annotations:
            attr_name = self.cat_id_to_attr.get(ann['category_id'])
            if attr_name and attr_name in targets:
                targets[attr_name] = int(ann['attributes'][attr_name])

        if self.transform:
            image = self._apply_transform(self.transform, image)

        if self.band_dir is not None:
            basename = os.path.splitext(image_filename)[0]
            band_path = os.path.join(self.band_dir, f"{basename}_band.jpg")
            band_image = Image.open(band_path).convert('RGB')
            band_image = self._apply_transform(self.band_transform, band_image)
            return image, band_image, targets

        return image, targets


# 测试数据读取
if __name__ == "__main__":
    dataset = DetectionDataset(
        annotation="/data/zhangxiaohao/dazhou/all_1_9/output/annotations/train_detection.json",
        root="/data/zhangxiaohao/dazhou/all_1_9/output/train/")
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=detection_collate_fn)
    # data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # 遍历数据集
    for images, targets in data_loader:
        print(images[0].shape)  # 图像形状
        print(targets)  # 每个目标字典
