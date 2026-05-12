import re
import matplotlib.pyplot as plt
import argparse


def parse_log(log_path):
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []
    val_accs = {'venous': [], 'nipple': [], 'arrangement': [], 'base_transparency': [], 'avg': []}

    with open(log_path, 'r') as f:
        for line in f:
            # Training Loss
            m = re.search(r'Epoch \[(\d+)/\d+\] - Training Loss: ([\d.]+)', line)
            if m:
                train_epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            # Validation Loss & Accuracies
            m = re.search(
                r'Epoch \[(\d+)/\d+\] - Validation Loss: ([\d.]+) '
                r'\| Venous Acc: ([\d.]+) '
                r'\| Nipple Acc: ([\d.]+) '
                r'\| Arrangement Acc: ([\d.]+) '
                r'\| BaseTransparency Acc: ([\d.]+) '
                r'\| Avg\. Val Acc: ([\d.]+)',
                line
            )
            if m:
                val_epochs.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_accs['venous'].append(float(m.group(3)))
                val_accs['nipple'].append(float(m.group(4)))
                val_accs['arrangement'].append(float(m.group(5)))
                val_accs['base_transparency'].append(float(m.group(6)))
                val_accs['avg'].append(float(m.group(7)))

    return train_epochs, train_losses, val_epochs, val_losses, val_accs


def plot(log_path, save_path):
    train_epochs, train_losses, val_epochs, val_losses, val_accs = parse_log(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Loss ---
    ax = axes[0]
    ax.plot(train_epochs, train_losses, label='Train Loss', linewidth=1.5)
    ax.plot(val_epochs, val_losses, label='Val Loss', linewidth=1.5, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    colors = {'venous': '#1f77b4', 'nipple': '#ff7f0e', 'arrangement': '#2ca02c', 'base_transparency': '#d62728', 'avg': '#9467bd'}
    for name, vals in val_accs.items():
        style = {'linewidth': 2.5, 'marker': 'o', 'markersize': 5} if name == 'avg' else {'linewidth': 1, 'alpha': 0.7}
        label = name if name != 'avg' else 'Avg. Val Acc'
        ax.plot(val_epochs, vals, label=label, color=colors[name], **style)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str,
                        default='/home/suzhiling/efficientnet/work_dir/models/classification/V5/logs/train_log_20260512_101746.txt')
    parser.add_argument('--save', type=str, default='./training_curves_V5.png')
    args = parser.parse_args()
    plot(args.log, args.save)
