"""Run the complete CPU experiment: collect, train, then evaluate sealed labels."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/perception/experiment')
    parser.add_argument('--train', type=int, default=64)
    parser.add_argument('--val', type=int, default=16)
    parser.add_argument('--test', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--motion-epochs', type=int, default=20)
    parser.add_argument('--segmentation-epochs', type=int, default=30)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--refine-labels', action='store_true')
    parser.add_argument('--preserve-motion', action='store_true')
    args = parser.parse_args()
    if args.preserve_motion and args.motion_epochs < 1:
        parser.error('--preserve-motion requires at least one motion training epoch')
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error('Choose an empty output directory; existing experiments are preserved.')
    from .collect import collect_dataset
    from .train import train
    from .evaluate import evaluate_checkpoint
    collect_dataset(root/'data', train=args.train, val=args.val, test=args.test, seed=args.seed)
    train(root/'data', root/'cache', root/'run', motion_epochs=args.motion_epochs,
          segmentation_epochs=args.segmentation_epochs, threads=args.threads, seed=args.seed,
          refine_labels=args.refine_labels)
    checkpoint = root/'run/model.pt'
    if args.preserve_motion:
        from .compose import compose_checkpoints
        checkpoint = root/'composed/model.pt'
        compose_checkpoints(root/'run/model.pt', root/'run/motion.pt', checkpoint)
    evaluate_checkpoint(data=root/'data', checkpoint=checkpoint,
                        output=root/'report', cache=root/'cache', threads=args.threads)


if __name__ == '__main__':
    main()
