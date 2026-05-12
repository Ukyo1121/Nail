# 甲襞微循环图像分类

基于 EfficientNet-B0 的多任务甲襞微循环（nail fold microcirculation）图像属性分类系统，同时完成 4 个属性的多级分类与回归。

## 项目结构

```
.
├── train_efficientnet.py    # 主训练脚本（多任务分类+回归）
├── eval_efficientnet.py      # 模型评估与可视化
├── train_nipple.py           # 消融实验：nipple 属性单独训练（原图 vs band）
├── eval_nipple_compare.py    # 消融评估：交叉对比原图 vs band 模式
├── edge_extract.py           # 甲襞上缘边界提取算法
├── preprocess_bands.py       # 批量 band 裁剪预处理
├── shape_tubes_dataset.py    # 数据加载（分类 / 检测 / 形态）
├── data_analyse.py           # 标签分布统计
├── plot_training_curves.py   # 训练曲线绘制
├── requirements.txt          # 依赖
├── bands/                    # band 裁剪图像（train / val）
├── work_dir/models/          # 模型权重与训练日志
├── vis_results/              # 评估可视化输出
└── edge_test_img/            # 边界提取调试输出
```

## 任务定义

模型同时对甲襞微循环图像的 4 个属性进行分类（有序类别）和回归：

| 属性 | 类别数 | 标签值 |
|---|---|---|
| Venous（静脉清晰度） | 4 | 0 / 1 / 2 / 3 |
| Nipple（乳头） | 4 | 0 / 1 / 2 / 3 |
| Arrangement（排列） | 4 | 0 / 1 / 2 / 3 |
| Base Transparency（基底透明度） | 3 | 0 / 1 / 2 |

## 环境

- Python >= 3.8
- PyTorch >= 1.12, torchvision >= 0.13
- 详见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 数据准备

数据采用 COCO 格式标注，目录结构：

```
数据根目录/
├── annotations/
│   ├── train_classification.json
│   └── val_classification.json
├── train/          # 训练图像 (*.jpg)
└── val/            # 验证图像 (*.jpg)
```

标注 JSON 中每个 category 的 `name` 需与属性名（`venous` / `nipple` / `arrangement` / `base_transparency`）一致，标签值存储在 `attributes` 字段中。

### 数据分布统计

```bash
python data_analyse.py
```

## 训练

```bash
python train_efficientnet.py \
    --train_ann /path/to/train_classification.json \
    --train_dir /path/to/train/images \
    --val_ann /path/to/val_classification.json \
    --val_dir /path/to/val/images \
    --batch_size 8 \
    --epochs 100 \
    --lr 5e-4 \
    --sigma 0.5 \
    --lambda_reg 0.3 \
    --patience 5 \
    --output_dir ./work_dir/models/classification/V8/
```

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--batch_size` | 8 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 5e-4 | 初始学习率 |
| `--sigma` | 0.5 | 排序感知损失的高斯 soft label σ |
| `--lambda_reg` | 0.3 | 回归辅助损失的权重 |
| `--patience` | 5 | 早停耐心值（评估周期） |
| `--eval_freq` | 4 | 每 N 个 epoch 评估一次 |

## 评估

```bash
python eval_efficientnet.py \
    --model_path ./work_dir/models/classification/V4/effiecientnet_classification_best.pth \
    --val_ann /path/to/val_classification.json \
    --val_dir /path/to/val/images \
    --vis_dir ./vis_results/
```

输出每属性的 accuracy、macro recall 以及各类别的 per-class recall。若指定 `--vis_dir`，会在原图上绘制 GT vs Pred 对比结果。

## 训练曲线

```bash
python plot_training_curves.py \
    --log work_dir/models/classification/V4/logs/train_log_xxx.txt \
    --save training_curves.png
```

## 边界提取（预处理）

在甲襞图像中自动检测上缘边界，用于提取 ROI（band 裁剪）：

```bash
# 单张调试
python edge_extract.py

# 批量预处理
python preprocess_bands.py \
    --input_dir /path/to/original/images \
    --output_dir ./bands/val_band \
    --band_width 200
```

算法流程：LAB 颜色空间 → 上下窗口颜色差异 → 抑制反光/血管区域 → 位置先验加权 → 动态规划最优路径。

## 消融实验

验证 band ROI 对 nipple 分类的提升效果：

```bash
# 训练（两种模式各一次）
python train_nipple.py --mode original --band_dir ./bands --output_dir ./work_dir/models/nipple_test_original
python train_nipple.py --mode band --band_dir ./bands --output_dir ./work_dir/models/nipple_test_band

# 交叉对比评估
python eval_nipple_compare.py \
    --original_model ./work_dir/models/nipple_test_original/nipple_original_best.pth \
    --band_model ./work_dir/models/nipple_test_band/nipple_band_best.pth
```

## 模型架构

```
EfficientNet-B0 (backbone, 不含 classifier)
    │
    ├── fc (1280 → 4)   × 4    # 分类头（venous / nipple / arrangement / base_transparency）
    └── reg (1280 → 1)  × 4    # 回归头（各属性连续值）
```

## 损失函数

- **分类**：Ordinal-aware Cross Entropy —— 用高斯分布替代 one-hot 作为 soft label，使模型对邻近类别的惩罚 < 对远处类别的惩罚
- **回归**：Smooth L1 Loss —— 将离散标签归一化到 [0,1] 后监督连续值
- **总损失** = 4 个分类损失之和 + λ × (4 个回归损失之和)

## 训练策略

- Optimizer: Adam, weight_decay=1e-5
- Scheduler: Cosine Annealing LR, eta_min=1e-7
- 早停: 基于平均验证准确率，patience=5 个评估周期
- 数据增强: RandomHorizontalFlip + ColorJitter（仅训练集）
