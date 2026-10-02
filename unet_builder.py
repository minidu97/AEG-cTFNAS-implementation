from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from operations import NORMAL_OPS, REDUCTION_OPS, UPSAMPLE_OPS
from genotype_decode import decode_encoder_block, decode_decoder_block, BlockConfig
from compact_encoding import ArchitectureGenotype


class NormalCell(nn.Module):
    def __init__(self, channels: int, config: BlockConfig):
        super().__init__()
        self.ops = nn.ModuleList([
            NORMAL_OPS[choice.op_idx](channels) for choice in config.normal_ops
        ])
        self.conn_idx = [choice.conn_idx for choice in config.normal_ops]

    def forward(self, x):
        outputs = [x]
        for op, conn in zip(self.ops, self.conn_idx):
            inp = outputs[conn]
            outputs.append(op(inp))
        cell_out = torch.stack(outputs[1:], dim=0).sum(dim=0)
        return cell_out + x


class ReductionCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: BlockConfig):
        super().__init__()
        self.ops = nn.ModuleList([
            REDUCTION_OPS[op_idx](in_channels) for op_idx in config.parallel_ops
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * len(self.ops), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = [op(x) for op in self.ops]
        return self.project(torch.cat(outs, dim=1))


class UpsampleCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: BlockConfig):
        super().__init__()
        self.ops = nn.ModuleList([
            UPSAMPLE_OPS[op_idx](in_channels) for op_idx in config.parallel_ops
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * len(self.ops), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = [op(x) for op in self.ops]
        return self.project(torch.cat(outs, dim=1))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: BlockConfig):
        super().__init__()
        self.normal = NormalCell(in_channels, config)
        self.reduce = ReductionCell(in_channels, out_channels, config)

    def forward(self, x):
        skip = self.normal(x)          # saved for the matching decoder block
        down = self.reduce(skip)
        return down, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 config: BlockConfig):
        super().__init__()
        self.up = UpsampleCell(in_channels, out_channels, config)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.normal = NormalCell(out_channels, config)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.fuse(torch.cat([x, skip], dim=1))
        return self.normal(x)


class SearchableUNet(nn.Module):
    def __init__(self, geno: ArchitectureGenotype, enc_samples, dec_samples,
                 in_channels: int = 1, num_classes: int = 4, base_channels: int = 16):
        super().__init__()
        n_blocks = len(geno.encoder_blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        enc_channels = [base_channels * (2 ** i) for i in range(n_blocks + 1)]
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            block_geno = geno.encoder_blocks[i]
            cfg = decode_encoder_block(enc_samples[i], block_geno.n_normal,
                                        block_geno.n_parallel)
            self.encoder_blocks.append(
                EncoderBlock(enc_channels[i], enc_channels[i + 1], cfg)
            )

        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            # mirror: first decoder block consumes the bottleneck
            block_geno = geno.decoder_blocks[i]
            cfg = decode_decoder_block(dec_samples[i], block_geno.n_normal,
                                        block_geno.n_parallel)
            in_ch = enc_channels[n_blocks - i]
            out_ch = enc_channels[n_blocks - i - 1]
            skip_ch = enc_channels[n_blocks - i - 1]
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, cfg)
            )

        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for block in self.encoder_blocks:
            x, skip = block(x)
            skips.append(skip)

        for i, block in enumerate(self.decoder_blocks):
            skip = skips[len(skips) - 1 - i]
            x = block(x, skip)

        return self.head(x)


def build_unet(geno: ArchitectureGenotype, enc_samples, dec_samples,
               in_channels: int = 1, num_classes: int = 4,
               base_channels: int = 16) -> SearchableUNet:
    """Convenience wrapper -- see SearchableUNet docstring."""
    return SearchableUNet(geno, enc_samples, dec_samples,
                           in_channels=in_channels, num_classes=num_classes,
                           base_channels=base_channels)
