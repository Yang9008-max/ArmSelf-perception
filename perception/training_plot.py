"""Export weak-supervision learning curves; their losses are not body accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training(exposure, controls, weighted, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for path, name, color, axis in ((exposure, 'Exposure only', '#3366bb', axes[0]),
                                     (controls, 'Controls as ordinary labels', '#cc7733', axes[0]),
                                     (weighted, 'Controls with separate weight', '#229966', axes[1])):
        rows = [row for row in json.loads(Path(path).read_text(encoding='utf-8')) if row['stage'] == 'segmentation']
        epochs = [row['epoch'] for row in rows]
        train, val = ([row[key][0] for row in rows] for key in ('train_loss', 'val_loss'))
        axis.plot(epochs, train, color=color, label=f'{name}: train')
        axis.plot(epochs, val, color=color, linestyle='--', label=f'{name}: validation')
        best = min(range(len(val)), key=val.__getitem__)
        axis.scatter([epochs[best]], [val[best]], color=color, marker='*', s=120, zorder=3)
    for axis in axes:
        axis.set_xlabel('Segmentation epoch')
        axis.set_ylabel('Observation-only objective')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_title('Original loss; different weak-label sets')
    axes[1].set_title('Additional evidence loss; separate scale')
    fig.suptitle('Training experience: validation loss selects weights, not body truth\nLoss magnitudes across objectives are not directly comparable', fontsize=11)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f'Saved {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exposure', 'controls', 'weighted', 'output'):
        parser.add_argument(f'--{name}', required=True)
    plot_training(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
