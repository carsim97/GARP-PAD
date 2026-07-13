import torch.nn as nn

from .encoder_r2 import R2Encoder
from .aggregator import PatchAggregator, MeanAggregator

class GARP_PAD(nn.Module):
    def __init__(self, embed_dim=64, invariant='normpool', aggregator='gated'):
        super().__init__()
        self.encoder = R2Encoder(embed_dim=embed_dim, invariant=invariant)
        if aggregator == 'gated':
            self.att = PatchAggregator(embed_dim=embed_dim)
        elif aggregator == 'mean':
            self.att = MeanAggregator(embed_dim=embed_dim)
        else:
            raise ValueError(f"unknown aggregator: {aggregator}")
        self.invariant = invariant
        self.aggregator = aggregator
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.encoder(x)
        pooled, A = self.att(emb)
        out = self.fc(pooled)
        return out, A
