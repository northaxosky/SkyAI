"""Score both trained checkpoints on an identical validation slice.

The in-run val_loss numbers are not comparable: eval.val_steps counts micro-batches, so
gpt2-muon (B=64, world=8) scored 10.49M tokens and skyai (B=32, world=4) only 2.62M.

    uv run python scripts/compare_val_loss.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

VAL = "data/edu_fineweb10B/edufineweb_val_000000.npy"
T = 1024
BATCH = 16
DEVICE = "cuda"


def load_model(cfg_name: str, path: str):
    """Build from the run's YAML; checkpoints store weights only."""
    from pathlib import Path

    from harness.config.loader import load_config
    from harness.training.loop import build_model

    cfg = load_config(Path(f"configs/{cfg_name}.yaml"), [])
    m = build_model(cfg.model)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(state, strict=True)
    return m.to(DEVICE).eval(), ck["step"], cfg.model.family


@torch.no_grad()
def per_sequence_losses(model, tokens: torch.Tensor, n_seq: int) -> np.ndarray:
    """Mean CE per sequence, so the two models can be compared pairwise."""
    out = np.empty(n_seq, dtype=np.float64)
    for start in range(0, n_seq, BATCH):
        n = min(BATCH, n_seq - start)
        idx = torch.stack([tokens[(start + i) * T : (start + i) * T + T + 1] for i in range(n)])
        x, y = idx[:, :-1].to(DEVICE), idx[:, 1:].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        ls = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        )
        out[start : start + n] = ls.view(n, -1).mean(dim=1).double().cpu().numpy()
    return out


def main() -> None:
    raw = np.load(VAL).astype(np.int32)
    tokens = torch.tensor(raw, dtype=torch.long)
    n_seq = (len(tokens) - 1) // T
    print(f"val shard: {len(raw):,} tokens -> {n_seq:,} sequences of {T}\n")

    results = {}
    for name in ("gpt2-muon", "skyai"):
        model, step, family = load_model(name, f"checkpoints/{name}/best.pt")
        losses = per_sequence_losses(model, tokens, n_seq)
        results[name] = losses
        print(f"{name:<10} family={family:<7} step={step} val_loss={losses.mean():.6f}")
        del model
        torch.cuda.empty_cache()

    a, b = results["gpt2-muon"], results["skyai"]
    d = a - b  # positive => skyai better

    print(f"\n{'window (sequences)':<24}{'gpt2-muon':>12}{'skyai':>12}{'gap':>12}")
    for k in (2560, 10240, n_seq):
        label = f"{k:,}" + (
            "  (skyai in-run)"
            if k == 2560
            else "  (gpt2-muon in-run)"
            if k == 10240
            else "  (FULL shard)"
        )
        print(
            f"{label:<24}{a[:k].mean():>12.6f}{b[:k].mean():>12.6f}{(a[:k] - b[:k]).mean():>+12.6f}"
        )

    # Block bootstrap: sequences within a document are correlated, so resample blocks.
    rng = np.random.default_rng(0)
    block = 512
    nblocks = len(d) // block
    trimmed = d[: nblocks * block].reshape(nblocks, block)
    boot = np.array([trimmed[rng.integers(0, nblocks, nblocks)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nFULL-shard paired gap: {d.mean():+.6f}")
    print(
        f"  moving-block bootstrap 95% CI: [{lo:+.6f}, {hi:+.6f}]  (block={block}, {nblocks} blocks)"
    )
    print(f"  skyai lower on {(d > 0).mean() * 100:.1f}% of {len(d):,} sequences")
    per_block = trimmed.mean(axis=1)
    print(
        f"  per-block gap range: [{per_block.min():+.6f}, {per_block.max():+.6f}], "
        f"{(per_block > 0).sum()}/{nblocks} blocks favour skyai"
    )


if __name__ == "__main__":
    main()
