import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fingerprint_dataset import FingerprintDataset
from preprocessing.preprocessor import Preprocessor
from models.garp_pad import GARP_PAD
from losses.focal_loss import FocalLoss


INPUT_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'


def run_train(args):
    num_patches = args.num_patches
    batch_size = args.batch_size

    ds = FingerprintDataset(os.path.join(INPUT_DIR, args.data_file))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=True
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    preprocessor = Preprocessor(patch_size=args.patch_size, num_patches=num_patches, device=device)

    model = GARP_PAD()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss().to(device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            data = preprocessor.process_batch(inputs)
            if data is None:
                continue

            preds, _ = model(data)
            loss = criterion(preds, labels.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] "
              f"loss={(running_loss / len(loader)):.4f}")

        scheduler.step()

    model.eval()
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, args.checkpoint_file))