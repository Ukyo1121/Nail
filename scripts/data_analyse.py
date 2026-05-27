from collections import Counter
from shape_tubes_dataset import ClassificationDataset
from torchvision import transforms

dataset = ClassificationDataset(
    annotation='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/annotations/train_classification.json',
    root='/data/zhangxiaohao/dazhouV2/Bclass/batch1/output/train',
    transform=transforms.ToTensor()
  )
counts = {'venous': Counter(), 'nipple': Counter(), 'arrangement': Counter(), 'base_transparency': Counter()}
for _, labels in dataset:
    for name in counts:
        v = labels[name]
        if v != -1:
            counts[name][v] += 1
for name, c in counts.items():
    print(f"{name}: {dict(sorted(c.items()))}")