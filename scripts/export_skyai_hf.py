"""Export a skyai (modern-family) checkpoint to a HuggingFace trust_remote_code repo.

skyai's stack (RMSNorm + post-embed norm, RoPE, SwiGLU, GQA, QK-norm with sharpening,
untied embeddings, logit soft-cap) does not match any built-in transformers architecture,
so the repo ships its own modeling code from scripts/hf_skyai/. Module names mirror the
training code, so weights transfer without key remapping; a logits-match check against the
harness model verifies that before anything is published.

    uv run python scripts/export_skyai_hf.py --checkpoint checkpoints/skyai/best.pt --out checkpoints/skyai-hf
    # then, with a write-scoped HF_TOKEN in .env:
    uv run python scripts/export_skyai_hf.py --checkpoint ... --out ... --repo <user>/skyai-modern-xs --push
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from harness.checkpoint import load_checkpoint
from harness.training.loop import build_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_HF_SRC = Path(__file__).resolve().parent / "hf_skyai"
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _load_hf_classes(out: Path):
    """Import the copied modeling code the same way the Hub will."""
    sys.path.insert(0, str(out.parent))
    pkg = out.name.replace("-", "_")
    staging = out.parent / pkg
    if staging != out:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(out, staging)
    (staging / "__init__.py").touch()
    mod = __import__(f"{pkg}.modeling_skyai", fromlist=["SkyAIForCausalLM"])
    cfg_mod = __import__(f"{pkg}.configuration_skyai", fromlist=["SkyAIConfig"])
    return cfg_mod.SkyAIConfig, mod.SkyAIForCausalLM, staging


def convert(checkpoint: Path, out: Path, dtype: str) -> None:
    bundle = load_checkpoint(checkpoint)
    mc = bundle.config.model
    if mc.family != "modern":
        raise SystemExit(f"export_skyai_hf only supports family=modern (got {mc.family!r})")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("configuration_skyai.py", "modeling_skyai.py"):
        shutil.copy2(_HF_SRC / name, out / name)

    SkyAIConfig, SkyAIForCausalLM, staging = _load_hf_classes(out)
    cfg = SkyAIConfig(
        vocab_size=mc.vocab_size,
        vocab_pad_multiple=mc.vocab_pad_multiple,
        n_layer=mc.n_layer,
        n_head=mc.n_head,
        n_kv_head=mc.n_kv_head,
        n_embd=mc.n_embd,
        hidden_multiple=mc.hidden_multiple,
        block_size=mc.block_size,
        rope_theta=mc.rope_theta,
        logit_softcap=mc.logit_softcap,
        tie_word_embeddings=mc.tie_weights,
    )
    cfg.auto_map = {
        "AutoConfig": "configuration_skyai.SkyAIConfig",
        "AutoModelForCausalLM": "modeling_skyai.SkyAIForCausalLM",
    }

    model = SkyAIForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(bundle.model_state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys (conversion bug): {unexpected}")
    # cos/sin are built in __init__ from rope_theta; the harness keeps them non-persistent.
    if set(missing) - {"cos", "sin"}:
        raise SystemExit(f"missing keys (conversion bug): {sorted(set(missing) - {'cos', 'sin'})}")

    model = model.to(_DTYPES[dtype]).eval()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(out)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"wrote HF model -> {out} (dtype={dtype})")


def verify(checkpoint: Path, out: Path) -> None:
    bundle = load_checkpoint(checkpoint)
    ours = build_model(bundle.config.model)
    ours.load_state_dict(bundle.model_state)
    ours.eval()

    _, SkyAIForCausalLM, staging = _load_hf_classes(out)
    hf = SkyAIForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()

    torch.manual_seed(0)
    idx = torch.randint(0, bundle.config.model.vocab_size, (2, 128))
    with torch.no_grad():
        ours_logits, _ = ours(idx)
        hf_logits = hf(idx).logits
    diff = (ours_logits.float() - hf_logits.float()).abs().max().item()
    agree = (ours_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    shutil.rmtree(staging, ignore_errors=True)

    print(f"verify: max|delta logits| = {diff:.3e}, argmax agreement = {agree:.4f}")
    if diff > 1e-4 or agree < 1.0:
        raise SystemExit("verify FAILED: HF port does not match the harness model")
    print("verify OK")


def push(out: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp32")
    ap.add_argument("--repo", type=str, default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    convert(args.checkpoint, args.out, args.dtype)
    verify(args.checkpoint, args.out)
    if args.push:
        if not args.repo:
            raise SystemExit("--push requires --repo <user>/<name>")
        push(args.out, args.repo)


if __name__ == "__main__":
    main()
