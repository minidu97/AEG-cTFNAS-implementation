from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from compact_encoding import S_N, S_R, S_U, NUM_NORMAL_OPS


@dataclass
class NormalOpChoice:
    op_idx: int    # which operation from S_N
    conn_idx: int  # 0 = block input, k>0 = output of the k-th earlier normal op


@dataclass
class BlockConfig:
    normal_ops: List[NormalOpChoice]   # length N_N
    parallel_ops: List[int]            # length N_R (encoder) or N_U (decoder)


def decode_normal_dim(x_c_val: float, position: int) -> NormalOpChoice:
    n_conn_choices = position + 1
    hi = S_N * n_conn_choices - 1
    # map [-1, 1] -> integer in [0, hi]
    val = int(np.clip(round((x_c_val + 1.0) / 2.0 * hi), 0, hi))
    op_idx = val % S_N
    conn_idx = val // S_N
    conn_idx = min(conn_idx, position)  # safety clamp
    return NormalOpChoice(op_idx=op_idx, conn_idx=conn_idx)


def decode_parallel_dim(x_c_val: float, pool_size: int) -> int:
    hi = pool_size - 1
    val = int(np.clip(round((x_c_val + 1.0) / 2.0 * hi), 0, hi))
    return val


def decode_block(x_c: np.ndarray, n_normal: int, n_parallel: int,
                  parallel_pool_size: int) -> BlockConfig:
    assert x_c.shape[0] == n_normal + n_parallel

    normal_ops = [
        decode_normal_dim(x_c[i], position=i) for i in range(n_normal)
    ]
    parallel_ops = [
        decode_parallel_dim(x_c[n_normal + j], parallel_pool_size)
        for j in range(n_parallel)
    ]
    return BlockConfig(normal_ops=normal_ops, parallel_ops=parallel_ops)


def decode_encoder_block(x_c: np.ndarray, n_normal: int, n_reduction: int) -> BlockConfig:
    return decode_block(x_c, n_normal, n_reduction, parallel_pool_size=S_R)


def decode_decoder_block(x_c: np.ndarray, n_normal: int, n_upsample: int) -> BlockConfig:
    return decode_block(x_c, n_normal, n_upsample, parallel_pool_size=S_U)
