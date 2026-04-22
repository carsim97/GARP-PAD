from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

class FingerprintDataset(Dataset):
    def __init__(self, samples_file):
        with open(samples_file, "r") as f:
            self.samples = [l.strip() for l in f.readlines() if l.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert('L')
        img = transforms(img)

        label = 1 if "live" in path.lower() else 0
        return img, torch.tensor(label)