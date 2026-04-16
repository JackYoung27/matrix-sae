
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Put the repo root on sys.path so `from core...` and `from experiments...`
# resolve when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.train import train as train_sae
from experiments.analysis.evaluate_downstream import (
    evaluate_downstream_perhead_matched,
    load_sae_from_checkpoint,
)
from experiments.extraction.extract_states import load_model_and_tokenizer

def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)

MODEL_DEFAULT = "Qwen/Qwen3.5-0.8B"
LAYER = 9
DEFAULT_N_FEATURES = 2048
DEFAULT_K = 32
SEED = 42
N_HEADS = 16

def _sweep_root_tag(sweep_kind: str, values: list[int], base_n_features: int, base_k: int) -> str:
    values_slug = "-".join(str(value) for value in values)
    return f"{sweep_kind}_{values_slug}_nf{base_n_features}_k{base_k}"

def _canonical_checkpoint_dir(
    matched_root: str,
    sae_type: str,
    layer: int,
    head: int,
    seed: int,
) -> Path:
    return Path(matched_root) / f"{sae_type}_L{layer}_H{head}_nf{DEFAULT_N_FEATURES}_k{DEFAULT_K}_s{seed}"

def _sensitivity_checkpoint_dir(
    sensitivity_root: str,
    sweep_root_tag: str,
    sae_type: str,
    layer: int,
    head: int,
    n_features: int,
    k: int,
    seed: int,
) -> Path:
    return (
        Path(sensitivity_root)
        / sweep_root_tag
        / f"{sae_type}_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
    )

def _load_existing_train_result(ckpt_dir: Path) -> dict[str, Any]:
    import torch

    best_path = ckpt_dir / "best.pt"
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    return {
        "best_mse": payload.get("val_mse"),
        "config": payload.get("config", {}),
    }

def train_sensitivity_head(
    sweep_root_tag: str,
    sae_type: str,
    head: int,
    states_dir: str,
    sensitivity_checkpoint_root: str,
    layer: int = LAYER,
    n_features: int = DEFAULT_N_FEATURES,
    k: int = DEFAULT_K,
    seed: int = SEED,
) -> dict[str, Any]:
    ckpt_dir = _sensitivity_checkpoint_dir(
        sensitivity_root=sensitivity_checkpoint_root,
        sweep_root_tag=sweep_root_tag,
        sae_type=sae_type,
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        seed=seed,
    )
    best_path = ckpt_dir / "best.pt"
    cfg_path = ckpt_dir / "config.json"

    if best_path.exists() and cfg_path.exists():
        existing = _load_existing_train_result(ckpt_dir)
        return {
            "layer": layer,
            "head": head,
            "sae_type": sae_type,
            "n_features": n_features,
            "k": k,
            "seed": seed,
            "best_mse": existing["best_mse"],
            "checkpoint_dir": str(ckpt_dir),
            "skipped": True,
        }

    print(f"Training {sae_type}, L{layer} H{head}, nf={n_features}, k={k}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type=sae_type,
        data_dir=states_dir,
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        lr=3e-4,
        lr_min=3e-5,
        batch_size=256,
        epochs=20,
        warmup_steps=50,
        norm_every=100,
        resample_every=250,
        rank=1,
        seed=seed,
        output_dir=str(ckpt_dir),
    )
    elapsed = time.time() - t0
    return {
        "layer": layer,
        "head": head,
        "sae_type": sae_type,
        "n_features": n_features,
        "k": k,
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": str(ckpt_dir),
        "skipped": False,
    }

def evaluate_sensitivity_downstream(
    states_dir: str,
    matched_checkpoint_root: str,
    sensitivity_checkpoint_root: str,
    output_dir: str,
    model_name: str,
    sweep_kind: str = "k",
    values_csv: str = "16,32,64",
    sae_types_csv: str = "flat,rank1,bilinear",
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import gc

    import numpy as np
    import torch

    values = _parse_int_csv(values_csv)
    sae_types = [part.strip() for part in sae_types_csv.split(",") if part.strip()]

    if sweep_kind not in {"k", "nf"}:
        raise ValueError(f"Unsupported sweep_kind={sweep_kind!r}")

    sweep_root_tag = _sweep_root_tag(
        sweep_kind=sweep_kind,
        values=values,
        base_n_features=DEFAULT_N_FEATURES,
        base_k=DEFAULT_K,
    )

    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")
    model.eval()

    corpus_ids = np.load(f"{states_dir}/corpus.npy", mmap_mode="r")
    actual = min(n_sequences, len(corpus_ids))
    batches = []
    batch_size = 8
    for start in range(0, actual, batch_size):
        end = min(start + batch_size, actual)
        batches.append(np.array(corpus_ids[start:end], dtype=np.int64))
    corpus_batches = [torch.tensor(batch, dtype=torch.long) for batch in batches]

    train_mse_summary: dict[str, dict[str, float | int | None]] = {}

    baseline_eval = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=corpus_batches,
        layer_idx=layer,
        sae_type_configs={},
        n_heads=N_HEADS,
        split_fraction=0.5,
        device="cuda",
    )

    result: dict[str, Any] = dict(baseline_eval)
    result["sae_results"] = {}

    for sae_type in sae_types:
        for value in values:
            if sweep_kind == "k":
                n_features = DEFAULT_N_FEATURES
                k = value
                tag = f"{sae_type}_k{value}"
                use_canonical = value == DEFAULT_K
            else:
                n_features = value
                k = DEFAULT_K
                tag = f"{sae_type}_nf{value}"
                use_canonical = value == DEFAULT_N_FEATURES

            head_saes: dict[int, tuple] = {}
            train_mses: list[float] = []
            for head in range(N_HEADS):
                if use_canonical:
                    ckpt_dir = _canonical_checkpoint_dir(
                        matched_root=matched_checkpoint_root,
                        sae_type=sae_type,
                        layer=layer,
                        head=head,
                        seed=seed,
                    )
                else:
                    ckpt_dir = _sensitivity_checkpoint_dir(
                        sensitivity_root=sensitivity_checkpoint_root,
                        sweep_root_tag=sweep_root_tag,
                        sae_type=sae_type,
                        layer=layer,
                        head=head,
                        n_features=n_features,
                        k=k,
                        seed=seed,
                    )

                sae, _, train_mse = load_sae_from_checkpoint(
                    str(ckpt_dir / "best.pt"),
                    str(ckpt_dir / "config.json"),
                    device="cuda",
                )
                head_saes[head] = (sae, sae_type)
                if train_mse is not None:
                    train_mses.append(float(train_mse))

            train_mse_summary[tag] = {
                "n_features": n_features,
                "k": k,
                "n_heads_loaded": len(head_saes),
                "mean_val_mse": _mean(train_mses),
            }

            single_result = evaluate_downstream_perhead_matched(
                model=model,
                tokenizer=tokenizer,
                corpus_batches=corpus_batches,
                layer_idx=layer,
                sae_type_configs={tag: head_saes},
                n_heads=N_HEADS,
                split_fraction=0.5,
                device="cuda",
                baseline_result=result["baseline"],
            )
            result["sae_results"][tag] = single_result["sae_results"][tag]

            for sae, _ in head_saes.values():
                sae.cpu()
            del head_saes
            gc.collect()
            torch.cuda.empty_cache()

    result["experiment"] = "downstream_sensitivity"
    result["sweep_kind"] = sweep_kind
    result["values"] = values
    result["seed"] = seed
    result["n_sequences"] = actual
    result["train_mse_summary"] = train_mse_summary

    out_dir = Path(output_dir) / "reviewer_experiments" / "downstream_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{sweep_root_tag}_L{layer}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

def main():
    parser = argparse.ArgumentParser(description="Downstream sensitivity sweep")
    parser.add_argument("--states-dir", required=True, help="Path to extracted states directory (replaces /data/states).")
    parser.add_argument("--output-dir", default="results/data", help="Directory for JSON outputs.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Root directory for SAE checkpoints (replaces /data/checkpoints).",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--model-cache-dir", default=None, help="HF cache dir (replaces /models).")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument("--sweep-kind", default="k", choices=["k", "nf"])
    parser.add_argument("--values", default="16,32,64", help="Comma-separated sweep values.")
    parser.add_argument("--sae-types", default="flat,rank1,bilinear")
    args = parser.parse_args()

    if args.model_cache_dir:
        import os

        os.environ.setdefault("HF_HOME", args.model_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.model_cache_dir)

    checkpoint_root = Path(args.checkpoint_dir)
    matched_checkpoint_root = str(checkpoint_root / "qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k")
    sensitivity_checkpoint_root = str(checkpoint_root / "review_downstream_sensitivity")

    t0 = time.time()
    values_list = _parse_int_csv(args.values)
    sae_type_list = [part.strip() for part in args.sae_types.split(",") if part.strip()]

    if args.sweep_kind not in {"k", "nf"}:
        raise ValueError(f"Unsupported sweep_kind={args.sweep_kind!r}")

    sweep_root_tag = _sweep_root_tag(
        sweep_kind=args.sweep_kind,
        values=values_list,
        base_n_features=DEFAULT_N_FEATURES,
        base_k=DEFAULT_K,
    )

    # Sequential. Parallelize with joblib/multiprocessing if desired.
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for sae_type in sae_type_list:
        for value in values_list:
            if args.sweep_kind == "k":
                n_features = DEFAULT_N_FEATURES
                k = value
                is_canonical = value == DEFAULT_K
            else:
                n_features = value
                k = DEFAULT_K
                is_canonical = value == DEFAULT_N_FEATURES

            if is_canonical:
                continue

            for head in range(N_HEADS):
                try:
                    result = train_sensitivity_head(
                        sweep_root_tag=sweep_root_tag,
                        sae_type=sae_type,
                        head=head,
                        states_dir=args.states_dir,
                        sensitivity_checkpoint_root=sensitivity_checkpoint_root,
                        layer=args.layer,
                        n_features=n_features,
                        k=k,
                        seed=args.seed,
                    )
                    train_results.append(result)
                    status = "skip" if result.get("skipped") else "done"
                    label = f"{args.sweep_kind}={value}"
                    print(
                        f"[{status}] {sae_type} {label} H{head}: "
                        f"MSE={result.get('best_mse'):.6e}"
                    )
                except Exception as exc:
                    failures.append(f"{sae_type} {args.sweep_kind}={value} H{head}: {exc}")
                    print(f"[fail] {sae_type} {args.sweep_kind}={value} H{head}: {exc}")

    if failures:
        raise RuntimeError("Downstream sensitivity training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluation...")
    downstream = evaluate_sensitivity_downstream(
        states_dir=args.states_dir,
        matched_checkpoint_root=matched_checkpoint_root,
        sensitivity_checkpoint_root=sensitivity_checkpoint_root,
        output_dir=args.output_dir,
        model_name=args.model,
        sweep_kind=args.sweep_kind,
        values_csv=args.values,
        sae_types_csv=args.sae_types,
        layer=args.layer,
        n_sequences=args.n_sequences,
        seed=args.seed,
    )

    summary = {
        "experiment": "downstream_sensitivity",
        "generated_at": time.time(),
        "sweep_kind": args.sweep_kind,
        "values": values_list,
        "layer": args.layer,
        "train_results": train_results,
        "downstream": downstream,
    }

    out_path = Path(args.output_dir) / f"downstream_sensitivity_{args.sweep_kind}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
