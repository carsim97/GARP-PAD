import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fingerprint_dataset import FingerprintDataset
from preprocessing.preprocessor import Preprocessor
from models.garp_pad import GARP_PAD


INPUT_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'


def run_eval(args):
    ds = FingerprintDataset(os.path.join(INPUT_DIR, args.data_file))
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # deterministic patch subsampling when a cap is used, so results are reproducible
    # and identical across ROI settings
    torch.manual_seed(0)

    preprocessor = Preprocessor(
        patch_size=args.patch_size, num_patches=None, device=device,
        roi_percentile=getattr(args, 'roi_percentile', 0.85),
        mask_ratio_thresh=getattr(args, 'mask_ratio', 0.8),
        max_eval_patches=getattr(args, 'max_eval_patches', None),
    )

    invariant = getattr(args, 'invariant', 'normpool')
    aggregator = getattr(args, 'aggregator', 'gated')

    model = GARP_PAD(invariant=invariant, aggregator=aggregator)
    model = model.to(device)

    state = torch.load(os.path.join(CHECKPOINT_DIR, args.checkpoint_file), map_location=device)
    model.load_state_dict(state)

    model.eval()

    correct = 0
    total = 0

    avg_time = 0.0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            start = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            data = preprocessor.process_batch(inputs)
            if data is None: continue

            preds, _ = model(data)

            probs = torch.sigmoid(preds)
            pred_labels = (probs > 0.5).long()

            stop = time.time()
            avg_time += stop - start

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            TP += ((pred_labels == 1) & (labels == 1)).sum().item()
            TN += ((pred_labels == 0) & (labels == 0)).sum().item()
            FP += ((pred_labels == 1) & (labels == 0)).sum().item()
            FN += ((pred_labels == 0) & (labels == 1)).sum().item()

    acc = correct / total
    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    bpcer = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    print("Accuracy:", acc)
    print("APCER:", apcer)
    print("BPCER:", bpcer)
    print("ACE:", (apcer + bpcer) / 2)

    avg_time /= len(loader)
    print("Inference time:", avg_time)