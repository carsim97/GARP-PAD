import torch
import torch.nn as nn


class PatchAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8, att_dim=128):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, embed_dim)

        self.att_a = nn.Linear(self.head_dim, att_dim)
        self.att_b = nn.Linear(self.head_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)

    def forward(self, x):
        B, P, D = x.shape

        x = self.in_proj(x)

        x = x.view(B, P, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)

        H = torch.tanh(self.att_a(x)) * torch.sigmoid(self.att_b(x))
        logits = self.att_w(H).squeeze(-1)
        A = torch.softmax(logits, dim=-1)

        out = torch.sum(A.unsqueeze(-1) * x, dim=2)

        out = out.reshape(B, D)

        return out, A.mean(dim=1)


class MeanAggregator(nn.Module):
    """Unweighted mean pooling over the per-patch descriptors (a_i = 1/n).

    Removes all attention parameters (V, U, w, heads); keeps D=64 and feeds the
    same downstream linear classifier.
    """

    def __init__(self, embed_dim, num_heads=8, att_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, P, D = x.shape
        out = x.mean(dim=1)
        A = x.new_full((B, P), 1.0 / P)
        return out, A