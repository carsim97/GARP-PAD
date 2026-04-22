import torch.nn as nn


class BaselineBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BaselineDWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                   padding=1, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)
        # Pointwise
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.bn_pw(self.pointwise(x))
        return x


class BaselineEncoder(nn.Module):
    def __init__(self, embed_dim=64, base_width=16):
        super().__init__()

        ch1 = 128
        ch2 = 256
        ch3 = 512

        self.stem = BaselineBlock(1, ch1)
        self.stage1 = BaselineDWBlock(ch1, ch2, stride=2)
        self.stage2 = BaselineDWBlock(ch2, ch3, stride=2)

        self.bottleneck = nn.Conv2d(ch3, base_width * 4, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 4, embed_dim)

    def forward(self, x):
        # x: (B, P, C, H, W)
        B, P, C, H, W = x.shape
        x = x.reshape(B * P, C, H, W)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.bottleneck(x)

        x = self.pool(x).view(B * P, -1)
        out = self.fc(x)

        return out.view(B, P, -1)