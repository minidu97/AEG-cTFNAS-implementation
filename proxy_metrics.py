from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn

try:
    from thop import profile as _thop_profile
except ImportError:
    _thop_profile = None


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
def compute_flops(model: nn.Module, input_shape) -> float:
    if _thop_profile is None:
        raise ImportError("pip install thop")
    model_copy = copy.deepcopy(model).to("cpu").eval()
    dummy = torch.randn(1, *input_shape)
    macs, _ = _thop_profile(model_copy, inputs=(dummy,), verbose=False)
    return float(macs) * 2.0


# ---------------------------------------------------------------------------
# jacob_cov
# ---------------------------------------------------------------------------
def compute_jacob_cov(model: nn.Module, inputs: torch.Tensor, eps: float = 1e-5) -> float:
    model.eval()
    inputs = inputs.clone().requires_grad_(True)
    model.zero_grad()

    output = model(inputs)
    # Single backward call for the whole batch at once (standard zero-cost
    # NAS implementation), NOT a Python loop with one backward() per
    # sample -- conv/batchnorm layers naturally keep per-sample gradients
    # correct in one pass since the batch dimension doesn't mix at the
    # input layer. A per-sample loop (an earlier version of this function
    # did that) is both much slower AND, on GPU, actively counterproductive:
    # many small sequential backward() calls incur per-call kernel-launch
    # overhead that dwarfs the actual compute, so it can end up SLOWER on
    # GPU than CPU. One vectorized call avoids all of that.
    output.backward(torch.ones_like(output))
    J = inputs.grad.detach().cpu().reshape(inputs.shape[0], -1).numpy()  # (N, D)
    # corrcoef needs variance > 0 per row; guard against degenerate rows
    if np.allclose(J.std(axis=1), 0):
        return float("-inf")  # totally uninformative architecture

    corrs = np.corrcoef(J)
    corrs = np.nan_to_num(corrs, nan=0.0)
    eigvals, _ = np.linalg.eig(corrs)
    eigvals = np.real(eigvals)
    score = -np.sum(np.log(np.abs(eigvals) + eps) + 1.0 / (np.abs(eigvals) + eps))
    return float(score)


# ---------------------------------------------------------------------------
# synflow
# ---------------------------------------------------------------------------
def compute_synflow(model: nn.Module, input_shape) -> float:
    model_copy = copy.deepcopy(model).to("cpu")
    model_copy.eval()
    model_copy = model_copy.double()

    signs = {}
    with torch.no_grad():
        for name, param in model_copy.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()

    model_copy.zero_grad()
    dummy = torch.ones([1] + list(input_shape)).double()
    output = model_copy(dummy)
    torch.sum(output).backward()

    score = 0.0
    for param in model_copy.parameters():
        if param.grad is not None:
            score += torch.sum(torch.abs(param * param.grad)).item()

    return float(score)


# ---------------------------------------------------------------------------
# Combination -> single scalar to MINIMIZE
# ---------------------------------------------------------------------------
def compute_proxy_scores(model: nn.Module, inputs: torch.Tensor,
                          device: str = "cpu") -> dict:
    c, h, w = inputs.shape[1:]
    flops = compute_flops(model, (c, h, w))
    synflow_raw = compute_synflow(model, (c, h, w))

    jacob_model = model.to(device)  # in-place move; fine since this model
                                     # is discarded after scoring anyway
    jacob = compute_jacob_cov(jacob_model, inputs)
    # synflow spans many orders of magnitude across architectures/depths;
    # log-space keeps the later z-score normalization well-behaved
    synflow_log = float(np.log1p(max(synflow_raw, 0.0)))
    return {"flops": flops, "jacob_cov": jacob, "synflow": synflow_log,
            "synflow_raw": synflow_raw}


class ProxyNormalizer:
    def __init__(self):
        self._stats = {k: {"mean": 0.0, "var": 1.0, "n": 0} for k in
                        ("flops", "jacob_cov", "synflow")}

    def _update(self, key: str, value: float):
        s = self._stats[key]
        s["n"] += 1
        delta = value - s["mean"]
        s["mean"] += delta / s["n"]
        s["var"] = max(s["var"] * (s["n"] - 1) / s["n"] + delta ** 2 / s["n"], 1e-8) \
            if s["n"] > 1 else 1.0

    def normalize(self, raw: dict) -> dict:
        out = {}
        for k in ("flops", "jacob_cov", "synflow"):
            v = raw[k]
            self._update(k, v)
            s = self._stats[k]
            std = max(s["var"] ** 0.5, 1e-8)
            out[k] = (v - s["mean"]) / std
        return out


def make_objective_fn(unet_kwargs: dict, sample_batch: torch.Tensor,
                       geno, weights=(1.0, 1.0, 1.0), device: str = "cpu"):
    from unet_builder import build_unet
    import numpy as np

    normalizer = ProxyNormalizer()
    n_blocks = len(geno.encoder_blocks)
    block_len = geno.encoder_blocks[0].mu.shape[0]  # dims per block
    sample_batch = sample_batch.to(device)

    def objective_fn(x_flat: np.ndarray) -> float:
        # split the flat vector back into per-block enc/dec sample arrays
        enc_samples = [x_flat[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
        offset = n_blocks * block_len
        dec_samples = [x_flat[offset + i * block_len: offset + (i + 1) * block_len]
                       for i in range(n_blocks)]

        model = build_unet(geno, enc_samples, dec_samples, **unet_kwargs)  # stays on CPU
        raw = compute_proxy_scores(model, sample_batch, device=device)
        z = normalizer.normalize(raw)

        w_flops, w_jacob, w_synflow = weights
        return w_flops * z["flops"] - w_jacob * z["jacob_cov"] - w_synflow * z["synflow"]

    return objective_fn
