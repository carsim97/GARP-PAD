import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn


class R2ConvBlock(nn.Module):
    def __init__(self, in_type, out_type, stride=1):
        super().__init__()

        self.conv = enn.R2Conv(in_type, out_type, 3, stride=stride, padding=1, bias=False)
        self.bn = enn.InnerBatchNorm(out_type)
        self.act = enn.ReLU(out_type, inplace=True)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class R2DWConvBlock(nn.Module):
    def __init__(self, in_type, out_type, stride=1, max_freq=1):
        super().__init__()

        self.depthwise = enn.R2Conv(
            in_type, in_type, kernel_size=3,
            stride=stride, padding=1,
            groups=len(in_type) // max_freq,
            bias=False
        )
        self.bn_dw = enn.InnerBatchNorm(in_type)
        self.act = enn.ReLU(in_type, inplace=True)

        self.pointwise = enn.R2Conv(
            in_type, out_type, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn_pw = enn.InnerBatchNorm(out_type)

    def forward(self, x):
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.bn_pw(self.pointwise(x))
        return x


class R2Encoder(nn.Module):
    def __init__(self, embed_dim=64, base_width=16):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)

        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        self.type1 = enn.FieldType(self.r2_act, base_width * [self.r2_act.regular_repr])
        self.type2 = enn.FieldType(self.r2_act, (base_width * 2) * [self.r2_act.regular_repr])
        self.type3 = enn.FieldType(self.r2_act, (base_width * 4) * [self.r2_act.regular_repr])

        self.stem = R2ConvBlock(self.in_type, self.type1)
        self.stage1 = R2DWConvBlock(self.type1, self.type2, stride=2)
        self.stage2 = R2DWConvBlock(self.type2, self.type3, stride=2)

        self.invariant_map = enn.NormPool(self.type3)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.final_ch = len(self.type3)
        self.fc = nn.Linear(self.final_ch, embed_dim)

    def forward(self, x):
        B, P, C, H, W = x.shape
        x = x.reshape(B * P, C, H, W)

        x = enn.GeometricTensor(x, self.in_type)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.invariant_map(x)

        t = x.tensor
        t = self.pool(t).view(B * P, -1)
        out = self.fc(t)

        return out.view(B, P, -1)
