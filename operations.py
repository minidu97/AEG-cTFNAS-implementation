from __future__ import annotations
import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                  dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Normal Cell pool (S_N = 5) -- channel-preserving, spatial-size-preserving
# ---------------------------------------------------------------------------
class NormalConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_bn_relu(channels, channels, 3)

    def forward(self, x):
        return self.op(x)


class NormalConv5x5(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_bn_relu(channels, channels, 5)

    def forward(self, x):
        return self.op(x)


class NormalDilatedConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_bn_relu(channels, channels, 3, dilation=2)

    def forward(self, x):
        return self.op(x)


class NormalDepthwiseSeparable(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1,
                                    groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class NormalIdentity(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x):
        return x


NORMAL_OPS = [
    NormalConv3x3,
    NormalConv5x5,
    NormalDilatedConv3x3,
    NormalDepthwiseSeparable,
    NormalIdentity,
]  # len == S_N == 5, keep in sync with compact_encoding.S_N


# ---------------------------------------------------------------------------
# Reduction Cell pool (S_R = 4) -- halves spatial size, channels preserved
# (an outer 1x1 conv handles the channel-count change after concatenation)
# ---------------------------------------------------------------------------
class ReducePoolMax(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.proj(self.pool(x))


class ReducePoolAvg(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.proj(self.pool(x))


class ReduceStridedConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_bn_relu(channels, channels, 3, stride=2)

    def forward(self, x):
        return self.op(x)


class ReduceStridedConv5x5(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_bn_relu(channels, channels, 5, stride=2)

    def forward(self, x):
        return self.op(x)


REDUCTION_OPS = [
    ReducePoolMax,
    ReducePoolAvg,
    ReduceStridedConv3x3,
    ReduceStridedConv5x5,
]  # len == S_R == 4


# ---------------------------------------------------------------------------
# Upsample Cell pool (S_U = 4) -- doubles spatial size, channels preserved
# (an outer 1x1 conv handles the channel-count change after concatenation)
# ---------------------------------------------------------------------------
class UpNearestConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.op = conv_bn_relu(channels, channels, 3)

    def forward(self, x):
        return self.op(self.up(x))


class UpBilinearConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.op = conv_bn_relu(channels, channels, 3)

    def forward(self, x):
        return self.op(self.up(x))


class UpTransposedConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.op(x)))


class UpPixelShuffle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.op = conv_bn_relu(channels, channels, 3)

    def forward(self, x):
        x = self.shuffle(self.expand(x))
        return self.op(x)


UPSAMPLE_OPS = [
    UpNearestConv,
    UpBilinearConv,
    UpTransposedConv,
    UpPixelShuffle,
]  # len == S_U == 4
