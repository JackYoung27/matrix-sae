
from __future__ import annotations

import argparse
import json
import re
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
from experiments.extraction.extract_states import (
    load_corpus_from_file,
    load_model_and_tokenizer,
    probe_state_dims,
)

def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]

def _positions_slug(positions: list[int]) -> str:
    return "-".join(str(pos) for pos in positions)

def _dataset_tag(layer: int, positions: list[int], n_per_position: int, seed: int) -> str:
    return f"L{layer}_pos{_positions_slug(positions)}_npp{n_per_position}_s{seed}"

def _model_is_large(model_name: str) -> bool:
    name = model_name.lower()
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)b\b", name):
        if float(match.group(1)) > 2.0:
            return True
    return False

def _extract_batch_size(model_name: str) -> int:
    return 8 if _model_is_large(model_name) else 32

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)

MODEL_DEFAULT = "Qwen/Qwen3.5-0.8B"
LAYER = 9
N_FEATURES = 2048
K = 32
SEED = 42
N_HEADS = 16

def _canonical_checkpoint_dir(matched_root: str, sae_type: str, layer: int, head: int, seed: int) -> Path:
    return Path(matched_root) / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"

def _mixed_dataset_root(mixed_data_root: str, dataset_tag: str) -> Path:
    return Path(mixed_data_root) / dataset_tag

def _mixed_checkpoint_dir(
    mixed_checkpoint_root: str,
    dataset_tag: str,
    sae_type: str,
    layer: int,
    head: int,
    seed: int,
) -> Path:
    return (
        Path(mixed_checkpoint_root)
        / dataset_tag
        / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"
    )

def _load_existing_train_result(ckpt_dir: Path) -> dict[str, Any]:
    import torch

    best_path = ckpt_dir / "best.pt"
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    return {
        "best_mse": payload.get("val_mse"),
        "config": payload.get("config", {}),
    }

def extract_temporal_states(
    states_dir: str,
    temporal_root_base: str,
    model_name: str,
    layer: int = LAYER,
    n_samples: int = 5000,
    positions_csv: str = "128,256,512,768,1024",
) -> dict[str, Any]:
    import numpy as np
    import torch

    positions = _parse_int_csv(positions_csv)

    states_dir_p = Path(states_dir)
    out_root = Path(temporal_root_base) / f"layer_{layer}"
    out_root.mkdir(parents=True, exist_ok=True)
    done_path = out_root / "temporal_metadata.json"

    if done_path.exists():
        existing = json.loads(done_path.read_text())
        if (
            existing.get("n_samples", 0) >= n_samples
            and existing.get("positions") == positions
            and existing.get("model") == model_name
            and existing.get("layer") == layer
        ):
            print("Temporal extraction already present with matching metadata. Skipping.")
            return existing

    model, tokenizer, _ = load_model_and_tokenizer(model_name, "cuda")
    model.eval()

    corpus_path = states_dir_p / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus at {corpus_path}")

    batch_size = _extract_batch_size(model_name)
    batches_full = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_samples)
    actual_samples = sum(batch.shape[0] for batch in batches_full)
    n_heads, key_dim, val_dim = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"Temporal extraction: {actual_samples} sequences, layer {layer}, {n_heads} heads")

    position_times: dict[int, float] = {}
    t0 = time.time()

    for pos in positions:
        pos_dir = out_root / f"pos_{pos}"
        pos_dir.mkdir(parents=True, exist_ok=True)
        pos_done = pos_dir / "done.json"
        if pos_done.exists():
            done = json.loads(pos_done.read_text())
            if done.get("n_samples") == actual_samples:
                print(f"  pos={pos}: already extracted, skipping")
                continue

        print(f"  Extracting position {pos}")
        tp0 = time.time()
        head_memmaps = []
        for head in range(n_heads):
            fpath = str(pos_dir / f"head_{head}.npy")
            head_memmaps.append(
                np.lib.format.open_memmap(
                    fpath,
                    mode="w+",
                    dtype=np.float16,
                    shape=(actual_samples, key_dim, val_dim),
                )
            )

        sample_offset = 0
        with torch.no_grad():
            for batch in batches_full:
                truncated = batch[:, :pos].to("cuda")
                bs = truncated.shape[0]
                outputs = model(input_ids=truncated, use_cache=True)
                cache = outputs.past_key_values
                state_np = cache.layers[layer].recurrent_states.float().cpu().numpy().astype(np.float16)
                for head in range(n_heads):
                    head_memmaps[head][sample_offset : sample_offset + bs] = state_np[:, head]
                sample_offset += bs
                del outputs, cache
                torch.cuda.empty_cache()

        for memmap in head_memmaps:
            memmap.flush()

        elapsed_pos = time.time() - tp0
        position_times[pos] = round(elapsed_pos, 1)
        pos_done.write_text(json.dumps({"layer": layer, "position": pos, "n_samples": sample_offset}))
        print(f"  pos={pos}: wrote {sample_offset} samples in {elapsed_pos:.1f}s")

    metadata = {
        "model": model_name,
        "layer": layer,
        "positions": positions,
        "n_samples": actual_samples,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "dtype": "float16",
        "extraction_time_s": round(time.time() - t0, 1),
        "per_position_time_s": position_times,
    }
    done_path.write_text(json.dumps(metadata, indent=2))
    return metadata

def build_mixed_position_dataset(
    temporal_root_base: str,
    mixed_data_root: str,
    layer: int = LAYER,
    positions_csv: str = "128,256,512,768,1024",
    n_per_position: int = 1000,
    selection_seed: int = SEED,
) -> dict[str, Any]:
    import numpy as np

    positions = _parse_int_csv(positions_csv)
    dataset_tag = _dataset_tag(layer=layer, positions=positions, n_per_position=n_per_position, seed=selection_seed)

    temporal_root = Path(temporal_root_base) / f"layer_{layer}"
    temporal_meta_path = temporal_root / "temporal_metadata.json"
    if not temporal_meta_path.exists():
        raise FileNotFoundError(f"Missing temporal metadata at {temporal_meta_path}")

    temporal_meta = json.loads(temporal_meta_path.read_text())
    available_positions = temporal_meta.get("positions", [])
    for pos in positions:
        if pos not in available_positions:
            raise ValueError(f"Position {pos} not present in extracted temporal data: {available_positions}")

    n_available = int(temporal_meta["n_samples"])
    n_heads = int(temporal_meta["n_heads"])
    key_dim = int(temporal_meta["key_head_dim"])
    val_dim = int(temporal_meta["value_head_dim"])

    if n_per_position > n_available:
        raise ValueError(f"Requested n_per_position={n_per_position}, but only {n_available} samples available")

    dataset_root = _mixed_dataset_root(mixed_data_root, dataset_tag)
    done_path = dataset_root / "metadata.json"
    if done_path.exists():
        existing = json.loads(done_path.read_text())
        if (
            existing.get("layer") == layer
            and existing.get("positions") == positions
            and existing.get("n_per_position") == n_per_position
            and existing.get("selection_seed") == selection_seed
        ):
            print(f"Mixed-position dataset already built: {dataset_tag}")
            return existing

    dataset_root.mkdir(parents=True, exist_ok=True)
    layer_dir = dataset_root / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(positions) * n_per_position
    selection_by_position: dict[int, list[int]] = {}
    for pos in positions:
        rng = np.random.default_rng(selection_seed + pos)
        chosen = np.sort(rng.choice(n_available, size=n_per_position, replace=False))
        selection_by_position[pos] = chosen.tolist()

    for head in range(n_heads):
        out_path = layer_dir / f"head_{head}.npy"
        out = np.lib.format.open_memmap(
            str(out_path),
            mode="w+",
            dtype=np.float16,
            shape=(total_samples, key_dim, val_dim),
        )

        offset = 0
        for pos in positions:
            src_path = temporal_root / f"pos_{pos}" / f"head_{head}.npy"
            src = np.load(src_path, mmap_mode="r")
            chosen = np.array(selection_by_position[pos], dtype=np.int64)
            out[offset : offset + n_per_position] = src[chosen]
            offset += n_per_position
        out.flush()

    metadata = {
        "source": "temporal_states",
        "layer": layer,
        "positions": positions,
        "n_per_position": n_per_position,
        "n_samples": total_samples,
        "selection_seed": selection_seed,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "dataset_tag": dataset_tag,
        "dtype": "float16",
    }
    done_path.write_text(json.dumps(metadata, indent=2))
    (layer_dir / "layer_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

def train_mixed_head(
    sae_type: str,
    head: int,
    dataset_tag: str,
    mixed_data_root: str,
    mixed_checkpoint_root: str,
    layer: int = LAYER,
    seed: int = SEED,
) -> dict[str, Any]:
    ckpt_dir = _mixed_checkpoint_dir(
        mixed_checkpoint_root=mixed_checkpoint_root,
        dataset_tag=dataset_tag,
        sae_type=sae_type,
        layer=layer,
        head=head,
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
            "seed": seed,
            "best_mse": existing["best_mse"],
            "checkpoint_dir": str(ckpt_dir),
            "skipped": True,
        }

    print(f"Training mixed-position {sae_type}, L{layer} H{head}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type=sae_type,
        data_dir=str(_mixed_dataset_root(mixed_data_root, dataset_tag)),
        layer=layer,
        head=head,
        n_features=N_FEATURES,
        k=K,
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
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": str(ckpt_dir),
        "skipped": False,
    }

def evaluate_mixed_position_downstream(
    dataset_tag: str,
    states_dir: str,
    matched_checkpoint_root: str,
    mixed_checkpoint_root: str,
    output_dir: str,
    model_name: str,
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import numpy as np
    import torch

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

    sae_type_configs: dict[str, dict[int, tuple]] = {}
    train_mse_summary: dict[str, dict[str, float | int | None]] = {}

    for sae_type in ["flat", "rank1", "bilinear"]:
        final_tag = f"{sae_type}_finaltoken"
        mixed_tag = f"{sae_type}_mixedpos"
        final_head_saes: dict[int, tuple] = {}
        mixed_head_saes: dict[int, tuple] = {}
        final_mses: list[float] = []
        mixed_mses: list[float] = []

        for head in range(N_HEADS):
            canonical_dir = _canonical_checkpoint_dir(
                matched_root=matched_checkpoint_root,
                sae_type=sae_type,
                layer=layer,
                head=head,
                seed=seed,
            )
            mixed_dir = _mixed_checkpoint_dir(
                mixed_checkpoint_root=mixed_checkpoint_root,
                dataset_tag=dataset_tag,
                sae_type=sae_type,
                layer=layer,
                head=head,
                seed=seed,
            )

            final_sae, _, final_mse = load_sae_from_checkpoint(
                str(canonical_dir / "best.pt"),
                str(canonical_dir / "config.json"),
                device="cuda",
            )
            mixed_sae, _, mixed_mse = load_sae_from_checkpoint(
                str(mixed_dir / "best.pt"),
                str(mixed_dir / "config.json"),
                device="cuda",
            )

            final_head_saes[head] = (final_sae, sae_type)
            mixed_head_saes[head] = (mixed_sae, sae_type)
            if final_mse is not None:
                final_mses.append(float(final_mse))
            if mixed_mse is not None:
                mixed_mses.append(float(mixed_mse))

        sae_type_configs[final_tag] = final_head_saes
        sae_type_configs[mixed_tag] = mixed_head_saes
        train_mse_summary[final_tag] = {"mean_val_mse": _mean(final_mses), "n_heads_loaded": len(final_head_saes)}
        train_mse_summary[mixed_tag] = {"mean_val_mse": _mean(mixed_mses), "n_heads_loaded": len(mixed_head_saes)}

    result = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=corpus_batches,
        layer_idx=layer,
        sae_type_configs=sae_type_configs,
        n_heads=N_HEADS,
        split_fraction=0.5,
        device="cuda",
    )
    result["experiment"] = "mixed_position_downstream"
    result["seed"] = seed
    result["dataset_tag"] = dataset_tag
    result["n_features"] = N_FEATURES
    result["k"] = K
    result["n_sequences"] = actual
    result["train_mse_summary"] = train_mse_summary

    out_dir = Path(output_dir) / "reviewer_experiments" / "mixed_position_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{dataset_tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

def main():
    parser = argparse.ArgumentParser(description="Mixed-position downstream experiment")
    parser.add_argument("--states-dir", required=True, help="Path to extracted states directory (replaces /data/states).")
    parser.add_argument("--output-dir", default="results/data", help="Directory for JSON outputs.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Root directory for SAE checkpoints (replaces /data/checkpoints).",
    )
    parser.add_argument(
        "--temporal-dir",
        default=None,
        help="Root for temporal state extractions (default: <checkpoint-dir>/../temporal_states).",
    )
    parser.add_argument(
        "--mixed-data-dir",
        default=None,
        help="Root for mixed-position datasets (default: <checkpoint-dir>/../states_mixed_position).",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--model-cache-dir", default=None, help="HF cache dir (replaces /models).")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--positions", default="128,256,512,768,1024")
    parser.add_argument("--extract-samples", type=int, default=5000)
    parser.add_argument("--n-per-position", type=int, default=1000)
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages to run: extract,build,train,evaluate (or 'all').",
    )
    args = parser.parse_args()

    if args.model_cache_dir:
        import os

        os.environ.setdefault("HF_HOME", args.model_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.model_cache_dir)

    checkpoint_root = Path(args.checkpoint_dir)
    matched_checkpoint_root = str(checkpoint_root / "qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k")
    mixed_checkpoint_root = str(checkpoint_root / "review_mixed_position")
    temporal_root_base = args.temporal_dir or str(checkpoint_root.parent / "temporal_states")
    mixed_data_root = args.mixed_data_dir or str(checkpoint_root.parent / "states_mixed_position")

    t0 = time.time()
    positions_list = _parse_int_csv(args.positions)
    dataset_tag = _dataset_tag(
        layer=args.layer,
        positions=positions_list,
        n_per_position=args.n_per_position,
        seed=args.seed,
    )

    selected_stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in selected_stages:
        selected_stages = {"extract", "build", "train", "evaluate"}

    summary: dict[str, Any] = {
        "experiment": "mixed_position_downstream",
        "generated_at": time.time(),
        "layer": args.layer,
        "positions": positions_list,
        "extract_samples": args.extract_samples,
        "n_per_position": args.n_per_position,
        "dataset_tag": dataset_tag,
    }

    if "extract" in selected_stages:
        print("Stage: extract temporal states")
        summary["temporal_extraction"] = extract_temporal_states(
            states_dir=args.states_dir,
            temporal_root_base=temporal_root_base,
            model_name=args.model,
            layer=args.layer,
            n_samples=args.extract_samples,
            positions_csv=args.positions,
        )

    if "build" in selected_stages:
        print("Stage: build mixed-position dataset")
        summary["dataset"] = build_mixed_position_dataset(
            temporal_root_base=temporal_root_base,
            mixed_data_root=mixed_data_root,
            layer=args.layer,
            positions_csv=args.positions,
            n_per_position=args.n_per_position,
            selection_seed=args.seed,
        )

    if "train" in selected_stages:
        print("Stage: train mixed-position SAEs")
        # Sequential. Parallelize with joblib/multiprocessing if desired.
        train_results: list[dict[str, Any]] = []
        failures: list[str] = []
        for sae_type in ["flat", "rank1", "bilinear"]:
            for head in range(N_HEADS):
                try:
                    result = train_mixed_head(
                        sae_type=sae_type,
                        head=head,
                        dataset_tag=dataset_tag,
                        mixed_data_root=mixed_data_root,
                        mixed_checkpoint_root=mixed_checkpoint_root,
                        layer=args.layer,
                        seed=args.seed,
                    )
                    train_results.append(result)
                    status = "skip" if result.get("skipped") else "done"
                    print(
                        f"[{status}] {sae_type} H{head}: "
                        f"MSE={result.get('best_mse'):.6e}"
                    )
                except Exception as exc:
                    failures.append(f"{sae_type} H{head}: {exc}")
                    print(f"[fail] {sae_type} H{head}: {exc}")

        if failures:
            raise RuntimeError("Mixed-position training failures: " + "; ".join(failures))
        summary["train_results"] = train_results

    if "evaluate" in selected_stages:
        print("Stage: evaluate downstream")
        summary["downstream"] = evaluate_mixed_position_downstream(
            dataset_tag=dataset_tag,
            states_dir=args.states_dir,
            matched_checkpoint_root=matched_checkpoint_root,
            mixed_checkpoint_root=mixed_checkpoint_root,
            output_dir=args.output_dir,
            model_name=args.model,
            layer=args.layer,
            n_sequences=args.n_sequences,
            seed=args.seed,
        )

    out_path = Path(args.output_dir) / "mixed_position_downstream.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
