# AEG-cTFNAS — unofficial reference implementation (WIP)

Starting-point implementation of the algorithm described in Section 2 of
*"Compact Training-free NAS with Alternating Evolution Game for Medical
Image Segmentation"* (Sun, Wang, Song — MICCAI 2025), built because the
authors' repo (https://github.com/spcity/AEG-cTFNAS) currently has no code.

This is **not** a verified reproduction — several details (exact operation
pools for Normal/Reduction/Upsample cells, the exact training-free proxy
combining FLOPs/jacob_cov/synflow, the precise X+E / X+D construction) are
under-specified in the paper text and had to be implemented as reasonable
interpretations. Treat this as a scaffold to test against your own UNet
backbone and to compare with the authors' code once they reply.

## What's implemented

| File | Paper section | Contents |
|---|---|---|
| `compact_encoding.py` | 2.1, Eq. 1 | Truncated-normal inverse-CDF sampling, block/architecture genotype |
| `agc.py` | 2.1 | Alternating Game Control: Eq. 2 contribution scores, adaptive `M` |
| `bayesian_inference.py` | 2.2 | Wasserstein distance per block, MAP logistic regression (Laplace approx.) for block contributions |
| `search_strategy.py` | 2.2, Eq. 3 | `Bestpool`, Eq. 3 candidate update, μ/σ update rule |
| `search_loop.py` | 2, Fig. 1 | Orchestrates all of the above into the full search loop |
| `demo.py` | — | Smoke test with a dummy objective (sum-of-squares) |

## What you still need to plug in

`search_loop.AEGcTFNASSearch` takes an `objective_fn(x_flat) -> float`.
You need to write the part the paper doesn't specify in reproducible
detail:

1. **Decode** the flat compact-encoding vector back into concrete
   operation choices (`compact_encoding.BlockGenotype.decode` gives you
   per-dimension indices — map these onto your actual Normal/Reduction/
   Upsample Cell operation pools, i.e. define what `S_N`, `S_R`, `S_U`
   concretely contain for your UNet).
2. **Build** the resulting UNet variant (PyTorch).
3. **Score** it with a training-free proxy — Section 3.1 says they combine
   FLOPs, `jacob_cov`, and `synflow` (both are standard, e.g. from the
   `zero-cost-nas` / NASLib literature) but doesn't give the exact
   combination formula, so you'll need to decide a weighting or replicate
   what's typical in the training-free NAS literature they cite ([1, 20]
   in the paper).
4. Return a single scalar to **minimize**.

Also worth double-checking against the authors' code once you get it:
- The exact shapes of `X+E` and `X+D` (this implementation constructs them
  by re-sampling one side's blocks while holding the other side's most
  recent sample fixed — the paper's wording is consistent with this but
  doesn't spell out the mechanics).
- Whether the "2 evaluations per iteration" budget claim means something
  more specific than what's implemented here (currently the demo loop
  evaluates 4 candidates per iteration for clarity/debuggability — see the
  `NOTE` comment in `search_loop.py` for how to trim it down to 2).

## Running the smoke test

```bash
pip install numpy scipy
python demo.py
```

This just confirms the encoding → AGC → Bayesian inference → Eq. 3 update
pipeline runs end-to-end and the best-pool score trends downward on a
toy (sum-of-squares) objective — it says nothing about segmentation
performance until you wire in a real UNet + proxy metric.

## Suggested next steps

1. Send the authors' code-request email (already sent) and keep checking
   the repo / your inbox.
2. Define your concrete `S_N` / `S_R` / `S_U` operation pools in
   `compact_encoding.py`.
3. Write `objective_fn` around your UNet builder + training-free proxy.
4. Compare your search trajectories/Table-3-style ablations against the
   paper's Fig. 3 and Table 3 once you have real numbers, and note any
   places your reimplementation and the original code (once received)
   diverge — that's useful to report back to your supervisor either way.
