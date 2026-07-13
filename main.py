import argparse
from scripts.train import run_train
from scripts.eval import run_eval


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    pt = sub.add_parser('train')
    pt.add_argument('--data_file', default='data.txt')
    pt.add_argument('--checkpoint_file', default='checkpoint.pth')
    pt.add_argument('--patch_size', type=int, default=32)
    pt.add_argument('--num_patches', type=int, default=2)
    pt.add_argument('--batch_size', type=int, default=128)
    pt.add_argument('--epochs', type=int, default=50)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--workers', type=int, default=4)
    pt.add_argument('--device', default='cuda')
    pt.add_argument('--invariant', choices=['normpool', 'maxpool'], default='normpool')
    pt.add_argument('--aggregator', choices=['gated', 'mean'], default='gated')

    pe = sub.add_parser('eval')
    pe.add_argument('--data_file', default='data.txt')
    pe.add_argument('--checkpoint_file', default='checkpoint.pth')
    pe.add_argument('--patch_size', type=int, default=32)
    pe.add_argument('--workers', type=int, default=4)
    pe.add_argument('--device', default='cuda')
    pe.add_argument('--invariant', choices=['normpool', 'maxpool'], default='normpool')
    pe.add_argument('--aggregator', choices=['gated', 'mean'], default='gated')
    pe.add_argument('--roi_percentile', type=float, default=0.85)
    pe.add_argument('--mask_ratio', type=float, default=0.8)
    pe.add_argument('--max_eval_patches', type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()

