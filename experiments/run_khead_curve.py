"""Progressive k-head replacement curve: replace k GDN heads at layer L with
SAE reconstructions, measure perplexity, and compare bilinear vs flat SAEs.

    python -m experiments.run_khead_curve --layer 9 --k-values 1,2,4,8,12,16 ...
or:
    python experiments/run_khead_curve.py --layer 9 ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from experiments.analysis.evaluate_downstream import (
    _patch_gdn_initial_states,
    load_sae_from_checkpoint,
    reconstruct_state_head,
)
from experiments.extraction.extract_states import (
    get_gdn_layer_indices,
    load_model_and_tokenizer,
)

def khead_replacement_curve(
    states_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    model: str,
    model_cache_dir: str | None = None,
    layer: int = 9,
    k_values: list[int] | None = None,
    sae_types: list[str] | None = None,
    n_subsets: int = 5,
    n_sequences: int = 200,
    seed: int = 42,
    checkpoint_source: str = "matched",
) -> dict[str, Any]:
    """Progressive k-head replacement: replace k heads, measure PPL.

    For each k, sample n_subsets random subsets of k heads.
    Compare bilinear vs flat scaling curves.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 12, 16]
    if sae_types is None:
        sae_types = ["rank1", "flat"]

    print(f"=== k-head replacement curve, L{layer}, source={checkpoint_source} ===")
    print(f"  k values: {k_values}, {n_subsets} subsets each, {n_sequences} sequences")
    print(f"  SAE types: {sae_types}")

    mdl, tokenizer, config = load_model_and_tokenizer(model, device="cuda")
    mdl.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_ids = np.load(str(states_dir / "corpus.npy"))
    n_seq = min(n_sequences, len(corpus_ids))
    seq_len = corpus_ids.shape[1]
    split_pos = seq_len // 2

    rng = np.random.default_rng(seed)

    results = {
        "layer": layer,
        "k_values": k_values,
        "sae_types": sae_types,
        "n_subsets": n_subsets,
        "checkpoint_source": checkpoint_source,
    }

    for sae_type in sae_types:
        print(f"\n  === {sae_type} ===")

        # Load all 16 per-head SAEs
        head_saes: dict[int, tuple[Any, str]] = {}
        for h in range(16):
            ckpt_dir = checkpoint_dir / f"{sae_type}_L{layer}_H{h}_nf2048_k32_s{seed}"
            best_path = ckpt_dir / "best.pt"
            if not best_path.exists():
                print(f"    missing checkpoint for H{h}: {best_path}")
                continue
            sae, cfg, _ = load_sae_from_checkpoint(
                str(best_path),
                str(ckpt_dir / "config.json") if (ckpt_dir / "config.json").exists() else None,
                device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type)

        if len(head_saes) < 16:
            print(f"  Only {len(head_saes)}/16 SAEs, skipping")
            continue

        curve: dict[int, dict[str, Any]] = {}
        for k in k_values:
            subset_ppls = []

            # Generate random head subsets
            if k == 16:
                subsets = [list(range(16))]  # only one way to pick all 16
            else:
                subsets = [sorted(rng.choice(16, size=k, replace=False).tolist())
                           for _ in range(n_subsets)]

            for subset_idx, head_subset in enumerate(subsets):
                total_loss = 0.0
                total_tokens = 0

                for seq_i in range(n_seq):
                    input_ids = torch.tensor(
                        corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda"
                    )
                    prefix = input_ids[:, :split_pos]
                    suffix = input_ids[:, split_pos:]

                    # Fresh prefix
                    prefix_out = mdl(input_ids=prefix, use_cache=True)
                    cache = prefix_out.past_key_values
                    gdn_states = {}
                    for idx in gdn_layers:
                        lc = cache.layers[idx]
                        if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                            gdn_states[idx] = lc.recurrent_states.clone()

                    # Replace only the k selected heads
                    state = gdn_states[layer]
                    for h in head_subset:
                        sae, stype = head_saes[h]
                        original = state[0, h].float()
                        reconstructed = reconstruct_state_head(sae, original, stype)
                        state[0, h] = reconstructed.to(state.dtype)

                    with _patch_gdn_initial_states(mdl, gdn_layers, gdn_states):
                        out = mdl(
                            input_ids=suffix, past_key_values=cache,
                            use_cache=False, labels=suffix,
                        )
                    n_tok = suffix.shape[1] - 1
                    total_loss += out.loss.item() * n_tok
                    total_tokens += n_tok

                ppl = math.exp(total_loss / total_tokens)
                subset_ppls.append(ppl)

                if k <= 4:
                    print(f"    k={k} subset {subset_idx} ({head_subset[:4]}...): PPL={ppl:.4f}")

            curve[k] = {
                "mean_ppl": float(np.mean(subset_ppls)),
                "std_ppl": float(np.std(subset_ppls)),
                "all_ppls": [float(p) for p in subset_ppls],
            }
            print(f"  k={k}: PPL={np.mean(subset_ppls):.4f} ± {np.std(subset_ppls):.4f}")

        results[sae_type] = curve

        # Free SAEs
        del head_saes
        torch.cuda.empty_cache()

    # Compute baseline (no replacement)
    print("\n  Computing baseline...")
    total_loss = 0.0
    total_tokens = 0
    for seq_i in range(n_seq):
        input_ids = torch.tensor(corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda")
        prefix = input_ids[:, :split_pos]
        suffix = input_ids[:, split_pos:]
        prefix_out = mdl(input_ids=prefix, use_cache=True)
        cache = prefix_out.past_key_values
        gdn_states = {}
        for idx in gdn_layers:
            lc = cache.layers[idx]
            if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                gdn_states[idx] = lc.recurrent_states.clone()
        with _patch_gdn_initial_states(mdl, gdn_layers, gdn_states):
            out = mdl(input_ids=suffix, past_key_values=cache, use_cache=False, labels=suffix)
        n_tok = suffix.shape[1] - 1
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    baseline_ppl = math.exp(total_loss / total_tokens)
    results["baseline_ppl"] = baseline_ppl
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Summary: compute deltas
    print("\n  === PPL delta (%) vs baseline ===")
    for sae_type in sae_types:
        if sae_type not in results:
            continue
        print(f"  {sae_type}:")
        for k in k_values:
            mean_ppl = results[sae_type][k]["mean_ppl"]
            delta = (mean_ppl - baseline_ppl) / baseline_ppl * 100
            print(f"    k={k:2d}: Δ={delta:+.3f}%")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"L{layer}_{checkpoint_source}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-dir", type=Path, required=True,
                        help="Directory containing corpus.npy")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the per-layer JSON result")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Root containing <sae_type>_L{L}_H{H}_nf2048_k32_s{seed} subdirs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-cache-dir", type=str, default=None,
                        help="Optional HF cache dir (sets HF_HOME if set)")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,12,16",
                        help="Comma-separated list of k values")
    parser.add_argument("--sae-types", type=str, default="rank1,flat",
                        help="Comma-separated SAE types to evaluate")
    parser.add_argument("--n-subsets", type=int, default=5)
    parser.add_argument("--checkpoint-source", type=str, default="matched",
                        choices=["matched", "ablation"])
    parser.add_argument("--local-output-json", type=Path, default=None,
                        help="Optional additional path to copy final JSON into")
    args = parser.parse_args()

    if args.model_cache_dir:
        import os
        os.environ.setdefault("HF_HOME", args.model_cache_dir)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    sae_types = [x.strip() for x in args.sae_types.split(",") if x.strip()]

    t0 = time.time()
    result = khead_replacement_curve(
        states_dir=args.states_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        model=args.model,
        model_cache_dir=args.model_cache_dir,
        layer=args.layer,
        k_values=k_values,
        sae_types=sae_types,
        n_subsets=args.n_subsets,
        n_sequences=args.n_sequences,
        seed=args.seed,
        checkpoint_source=args.checkpoint_source,
    )

    print(f"\nDone in {time.time() - t0:.0f}s")

    if args.local_output_json is not None:
        args.local_output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.local_output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {args.local_output_json}")

if __name__ == "__main__":
    main()
