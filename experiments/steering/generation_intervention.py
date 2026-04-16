#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Text statistics

def compute_generation_stats(text: str) -> dict[str, float]:
    """Compute document-level statistics from generated text."""
    n_chars = max(len(text), 1)

    # Sentences: split on sentence-ending punctuation followed by space or end
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    mean_sentence_length = sum(len(s) for s in sentences) / n_sentences

    n_periods = text.count(".")
    n_newlines = text.count("\n")
    n_paragraphs = text.count("\n\n") + 1

    words = text.split()
    n_words = max(len(words), 1)
    mean_word_length = sum(len(w) for w in words) / n_words if words else 0.0

    return {
        "n_sentences": float(n_sentences),
        "mean_sentence_length": mean_sentence_length,
        "n_periods": float(n_periods),
        "n_newlines": float(n_newlines),
        "n_paragraphs": float(n_paragraphs),
        "period_density": n_periods / n_chars * 100,
        "newline_density": n_newlines / n_chars * 100,
        "mean_word_length": mean_word_length,
        "n_words": float(n_words),
    }

# Feature selection from probe results

BOUNDARY_KEYWORDS = ("sentence", "boundary", "paragraph", "period", "newline", "sent")

def select_boundary_features_fast(
    sae,
    sae_type: str,
    states_at_boundary: np.ndarray,
    states_at_nonboundary: np.ndarray,
    n_features: int = 10,
) -> list[dict[str, Any]]:
    """Find features that activate differently at boundary vs non-boundary positions.

    Encode both sets through the SAE, compute mean activation per feature,
    return top n_features by absolute mean difference.

    Args:
        sae: trained SAE model (on GPU, eval mode)
        sae_type: "flat", "rank1", "bilinear", or "bilinear_tied"
        states_at_boundary: (N, d_k, d_v) array of states at boundary positions
        states_at_nonboundary: (M, d_k, d_v) array of states at non-boundary positions
        n_features: how many top features to return

    Returns list of dicts with feature_idx, mean_boundary, mean_nonboundary,
    mean_boundary_nonzero, mean_nonboundary_nonzero, alive fractions, and mean_diff.
    """
    device = next(sae.parameters()).device

    def _encode_batch(states_np: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Encode states through SAE in batches, return activations as numpy."""
        all_coeffs = []
        for start in range(0, len(states_np), batch_size):
            batch = torch.tensor(
                states_np[start : start + batch_size], dtype=torch.float32, device=device,
            )
            if sae_type == "flat":
                batch = batch.reshape(batch.shape[0], -1)
            coeffs = sae.encode(batch)
            all_coeffs.append(coeffs.detach().cpu().numpy())
        return np.concatenate(all_coeffs, axis=0)

    coeffs_boundary = _encode_batch(states_at_boundary)
    coeffs_nonboundary = _encode_batch(states_at_nonboundary)

    mean_b = coeffs_boundary.mean(axis=0)
    mean_nb = coeffs_nonboundary.mean(axis=0)
    mean_diff = mean_b - mean_nb
    boundary_alive = coeffs_boundary > 0
    nonboundary_alive = coeffs_nonboundary > 0
    mean_b_nonzero = np.divide(
        coeffs_boundary.sum(axis=0),
        np.maximum(boundary_alive.sum(axis=0), 1),
        out=np.zeros_like(mean_b, dtype=float),
        where=np.ones_like(mean_b, dtype=bool),
    )
    mean_nb_nonzero = np.divide(
        coeffs_nonboundary.sum(axis=0),
        np.maximum(nonboundary_alive.sum(axis=0), 1),
        out=np.zeros_like(mean_nb, dtype=float),
        where=np.ones_like(mean_nb, dtype=bool),
    )
    mean_b_nonzero = np.where(boundary_alive.any(axis=0), mean_b_nonzero, 0.0)
    mean_nb_nonzero = np.where(nonboundary_alive.any(axis=0), mean_nb_nonzero, 0.0)
    alive_frac_b = boundary_alive.mean(axis=0)
    alive_frac_nb = nonboundary_alive.mean(axis=0)

    # Rank by absolute difference
    top_indices = np.argsort(-np.abs(mean_diff))[:n_features]
    results = []
    for idx in top_indices:
        idx = int(idx)
        results.append({
            "feature_idx": idx,
            "mean_boundary": float(mean_b[idx]),
            "mean_nonboundary": float(mean_nb[idx]),
            "mean_boundary_nonzero": float(mean_b_nonzero[idx]),
            "mean_nonboundary_nonzero": float(mean_nb_nonzero[idx]),
            "alive_frac_boundary": float(alive_frac_b[idx]),
            "alive_frac_nonboundary": float(alive_frac_nb[idx]),
            "mean_diff": float(mean_diff[idx]),
        })
    return results

def select_boundary_features(
    probe_results_path: str,
    n_features: int = 10,
    target_properties: list[str] | None = None,
) -> dict[str, Any]:
    """Load probe results and find features most correlated with boundary properties.

    Returns dict with:
      - per_property: {property_name -> [{feature_idx, rho, p_value}, ...]}
      - combined: deduplicated list of top features across all boundary properties
    """
    with open(probe_results_path) as f:
        data = json.load(f)

    features_list = data.get("probe", {}).get("features", [])
    if not features_list:
        raise ValueError(f"No features found in {probe_results_path}")

    # Determine which text properties are boundary-related
    if target_properties is None:
        # Collect all property names from the first feature's correlations
        all_props = set()
        for feat in features_list:
            all_props.update(feat.get("all_correlations", {}).keys())
        target_properties = [
            p for p in sorted(all_props)
            if any(kw in p.lower() for kw in BOUNDARY_KEYWORDS)
        ]

    per_property: dict[str, list[dict]] = {}
    for prop in target_properties:
        scored = []
        for feat in features_list:
            corr = feat.get("all_correlations", {}).get(prop)
            if corr is None:
                continue
            scored.append({
                "feature_idx": feat["feature_idx"],
                "rho": corr["rho"],
                "p_value": corr["p"],
            })
        scored.sort(key=lambda x: abs(x["rho"]), reverse=True)
        per_property[prop] = scored[:n_features]

    # Combine: deduplicate by feature_idx, keep highest |rho| entry
    seen: dict[int, dict] = {}
    for prop, entries in per_property.items():
        for entry in entries:
            fidx = entry["feature_idx"]
            if fidx not in seen or abs(entry["rho"]) > abs(seen[fidx]["rho"]):
                seen[fidx] = {**entry, "property": prop}
    combined = sorted(seen.values(), key=lambda x: abs(x["rho"]), reverse=True)
    combined = combined[:n_features]

    return {
        "per_property": per_property,
        "combined": combined,
        "target_properties": target_properties,
    }

# Core generation with per-step SAE intervention

@torch.no_grad()
def generate_with_intervention(
    model,
    tokenizer,
    sae=None,
    sae_type: str = "bilinear",
    layer_idx: int = 9,
    head_idx: int | None = None,
    prompt_ids: torch.Tensor | None = None,
    feature_updates: dict[int, float] | None = None,
    n_tokens: int = 200,
    temperature: float = 0.7,
    *,
    sae_per_head: dict[int, Any] | None = None,
    feature_updates_per_head: dict[int, dict[int, float]] | None = None,
    additive: bool = False,
    additive_min_zero: bool = False,
) -> tuple[list[int], dict]:
    """Generate n_tokens with continuous SAE feature intervention at every step.

    Supports two calling conventions:

    Single-head (backward compatible):
      sae, head_idx, feature_updates -- intervene on one head.

    Multi-head:
      sae_per_head={head_idx: sae, ...},
      feature_updates_per_head={head_idx: {feat: scale}, ...}
      -- intervene on multiple heads simultaneously.

    When additive=False (default), feature_updates maps feat_idx -> scale_factor
    and the intervention is multiplicative: coeffs[fi] *= scale.
    This compounds through the recurrence.

    When additive=True, feature_updates maps feat_idx -> additive_push and
    the intervention is: coeffs[fi] += push. This does NOT compound because
    each step adds a fixed delta regardless of the current activation magnitude.
    When additive_min_zero=True, edited coefficients are clamped at zero after the
    additive update. This gives suppression semantics that match the nonnegative
    TopK+ReLU SAE code space.

    At each step:
      1. Forward one token through the model with use_cache=True
      2. For each target head: extract state, encode through SAE,
         apply feature updates, decode, patch back
      3. Sample next token from logits with temperature

    Returns (generated_token_ids, metadata_dict).
    """
    # Normalize single-head args into multi-head dicts
    if sae_per_head is None:
        if sae is None:
            raise ValueError("Provide either sae or sae_per_head")
        sae_per_head = {head_idx: sae}
        feature_updates_per_head = {head_idx: feature_updates or {}}
    if feature_updates_per_head is None:
        feature_updates_per_head = {}

    device = prompt_ids.device

    # Step 1: process entire prompt to build initial cache
    prompt_out = model(input_ids=prompt_ids, use_cache=True)
    cache = prompt_out.past_key_values
    logits = prompt_out.logits[:, -1, :]  # (1, vocab)

    generated_ids: list[int] = []
    n_interventions = 0
    intervention_norms: list[float] = []

    # Pre-compute sorted feature indices per head for fast inner loop
    # Values are scale factors (multiplicative) or push magnitudes (additive)
    head_fi_updates: dict[int, tuple[list[int], list[float]]] = {}
    for h, updates in feature_updates_per_head.items():
        if updates:
            fi_sorted = sorted(updates.keys())
            head_fi_updates[h] = (fi_sorted, [updates[fi] for fi in fi_sorted])

    for step in range(n_tokens):
        # Sample next token (with NaN guard for aggressive interventions)
        logits_f = logits.float()
        if torch.isnan(logits_f).any() or torch.isinf(logits_f).any():
            logits_f = torch.zeros_like(logits_f)  # uniform fallback
        probs = F.softmax(logits_f / temperature, dim=-1)
        probs = probs.clamp(min=0.0)  # guard against negative from numerical issues
        if probs.sum() < 1e-8:
            probs = torch.ones_like(probs) / probs.shape[-1]
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        token_id = int(next_token[0, 0].item())

        generated_ids.append(token_id)

        # Forward the new token
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]

        # Intervene on each target head
        if head_fi_updates:
            layer_cache = cache.layers[layer_idx]
            state = layer_cache.recurrent_states  # (1, n_heads, d_k, d_v)
            cache_dtype = state.dtype

            for h_idx, (fi_list, update_list) in head_fi_updates.items():
                head_sae = sae_per_head[h_idx]
                original_head = state[0, h_idx].float()  # (d_k, d_v)

                # Encode
                if sae_type == "flat":
                    x = original_head.reshape(1, -1)
                else:
                    x = original_head.unsqueeze(0)  # (1, d_k, d_v)
                coeffs = head_sae.encode(x).clone()

                # Compute SAE residual (what the SAE cannot represent)
                recon_original = head_sae._decode(coeffs)
                if sae_type == "flat":
                    residual = x - recon_original  # (1, d_in)
                else:
                    residual = x - recon_original  # (1, d_k, d_v)

                # Apply feature updates
                if additive:
                    for fi, push in zip(fi_list, update_list):
                        coeffs[0, fi] += push
                    if additive_min_zero:
                        coeffs.clamp_(min=0.0)
                else:
                    for fi, scale in zip(fi_list, update_list):
                        coeffs[0, fi] *= scale

                # Decode modified coefficients and add back residual
                recon_modified = head_sae._decode(coeffs)
                recon_with_residual = recon_modified + residual
                if sae_type == "flat":
                    new_head = recon_with_residual.reshape(original_head.shape)
                else:
                    new_head = recon_with_residual.squeeze(0)

                # Track intervention magnitude
                delta_norm = float((new_head - original_head).norm().item())
                intervention_norms.append(delta_norm)

                # Patch back
                state[0, h_idx] = new_head.to(cache_dtype)

            n_interventions += 1

    metadata = {
        "n_generated": len(generated_ids),
        "n_interventions": n_interventions,
        "n_heads_intervened": len(head_fi_updates),
        "mean_intervention_norm": float(np.mean(intervention_norms)) if intervention_norms else 0.0,
        "stopped_early": len(generated_ids) < n_tokens,
    }
    return generated_ids, metadata

# Experiment runner

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (x - y)."""
    diff = x - y
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff < 1e-12:
        return 0.0
    return mean_diff / std_diff

def run_generation_experiment(
    model,
    tokenizer,
    sae=None,
    sae_type: str = "bilinear",
    layer_idx: int = 9,
    head_idx: int | None = None,
    prompts: list[str] | None = None,
    boundary_features: list[int] | None = None,
    boost_scale: float = 3.0,
    suppress_scale: float = 0.0,
    n_tokens: int = 200,
    temperature: float = 0.7,
    *,
    sae_per_head: dict[int, Any] | None = None,
    boundary_features_per_head: dict[int, list[int]] | None = None,
    additive: bool = False,
    additive_push_per_head: dict[int, dict[int, float]] | None = None,
    additive_boost_strength: float = 0.5,
    additive_min_zero: bool = False,
) -> dict:
    """Run the full generation intervention experiment.

    Supports single-head (sae, head_idx, boundary_features) or multi-head
    (sae_per_head, boundary_features_per_head) calling conventions.

    When additive=False (default, multiplicative mode):
      For each prompt, generate under 3 conditions:
        1. baseline: no feature modification
        2. boost: boundary features *= boost_scale
        3. suppress: boundary features *= suppress_scale

    When additive=True:
      feature_updates map feat_idx -> additive_push (constant per step).
      additive_push_per_head provides the push values per head per feature.
      boost condition: coeffs[fi] += push
      suppress condition: coeffs[fi] -= push
      No compounding through the recurrence.
      When additive_min_zero=True, additive edits are clamped at zero afterward.

    Returns aggregate statistics with paired t-tests and Cohen's d.
    """
    from scipy import stats

    # Normalize single-head into multi-head dicts
    if sae_per_head is None:
        sae_per_head = {head_idx: sae}
    if boundary_features_per_head is None:
        if boundary_features is not None:
            boundary_features_per_head = {head_idx: boundary_features}
        else:
            boundary_features_per_head = {}

    if additive:
        # Build additive updates per head
        # additive_push_per_head: {head_idx: {feat_idx: push_value}}
        if additive_push_per_head is None:
            raise ValueError("additive=True requires additive_push_per_head")

        def _build_updates_additive(sign: float) -> dict[int, dict[int, float]]:
            if sign == 0.0:
                return {}
            return {
                h: {f: sign * additive_push_per_head[h][f] for f in feats if f in additive_push_per_head.get(h, {})}
                for h, feats in boundary_features_per_head.items()
                if feats
            }

        conditions_updates = {
            "baseline": {},
            "boost": _build_updates_additive(+1.0),
            "suppress": _build_updates_additive(-1.0),
        }
    else:
        # Build per-head feature_updates for each condition (multiplicative)
        def _build_updates(scale: float) -> dict[int, dict[int, float]]:
            if scale == 1.0:
                return {}
            return {
                h: {f: scale for f in feats}
                for h, feats in boundary_features_per_head.items()
                if feats
            }

        conditions_updates = {
            "baseline": {},  # empty = no intervention
            "boost": _build_updates(boost_scale),
            "suppress": _build_updates(suppress_scale),
        }

    device = next(model.parameters()).device
    stat_names = list(compute_generation_stats("test text.").keys())
    per_prompt_stats: dict[str, list[dict[str, float]]] = {c: [] for c in conditions_updates}
    per_prompt_details: list[dict] = []

    heads_list = sorted(sae_per_head.keys())
    all_boundary_features = {
        h: boundary_features_per_head.get(h, []) for h in heads_list
    }

    t0 = time.time()
    for i, prompt_text in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prompt_detail: dict[str, Any] = {"prompt_idx": i, "prompt_len": int(prompt_ids.shape[1])}

        for cond_name, updates_per_head in conditions_updates.items():
            gen_ids, meta = generate_with_intervention(
                model=model,
                tokenizer=tokenizer,
                sae_type=sae_type,
                layer_idx=layer_idx,
                prompt_ids=prompt_ids,
                n_tokens=n_tokens,
                temperature=temperature,
                sae_per_head=sae_per_head if updates_per_head else {},
                feature_updates_per_head=updates_per_head,
                additive=additive,
                additive_min_zero=additive_min_zero,
            )
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_stats = compute_generation_stats(gen_text)
            per_prompt_stats[cond_name].append(gen_stats)
            prompt_detail[cond_name] = {
                "text": gen_text[:500],
                "stats": gen_stats,
                "n_generated": meta["n_generated"],
                "mean_intervention_norm": meta["mean_intervention_norm"],
            }

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(prompts) - i - 1) / rate if rate > 0 else 0
        print(
            f"  [{i+1}/{len(prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining, "
            f"baseline_words={per_prompt_stats['baseline'][-1]['n_words']:.0f}, "
            f"boost_words={per_prompt_stats['boost'][-1]['n_words']:.0f}, "
            f"suppress_words={per_prompt_stats['suppress'][-1]['n_words']:.0f}",
            flush=True,
        )
        per_prompt_details.append(prompt_detail)

    total_time = time.time() - t0

    # Aggregate: mean per condition
    condition_means: dict[str, dict[str, float]] = {}
    for cond_name in conditions_updates:
        means = {}
        for stat in stat_names:
            vals = [s[stat] for s in per_prompt_stats[cond_name]]
            means[stat] = float(np.mean(vals))
        condition_means[cond_name] = means

    # Statistical tests: paired t-test + Cohen's d
    test_results: dict[str, dict[str, dict[str, float | str]]] = {}
    for comparison, cond_a, cond_b in [
        ("boost_vs_baseline", "boost", "baseline"),
        ("suppress_vs_baseline", "suppress", "baseline"),
    ]:
        comp = {}
        for stat in stat_names:
            vals_a = np.array([s[stat] for s in per_prompt_stats[cond_a]])
            vals_b = np.array([s[stat] for s in per_prompt_stats[cond_b]])
            t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
            d = _cohens_d(vals_a, vals_b)
            direction = "higher" if float(np.mean(vals_a)) > float(np.mean(vals_b)) else "lower"
            comp[stat] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": d,
                "direction": direction,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
            }
        test_results[comparison] = comp

    result = {
        "experiment": "generation_intervention_multihead" if len(heads_list) > 1 else "generation_intervention",
        "layer": layer_idx,
        "heads": heads_list,
        "boundary_features_per_head": {str(h): feats for h, feats in all_boundary_features.items()},
        "n_heads": len(heads_list),
        "n_prompts": len(prompts),
        "n_tokens": n_tokens,
        "boost_scale": boost_scale,
        "suppress_scale": suppress_scale,
        "additive": additive,
        "additive_min_zero": additive_min_zero,
        "temperature": temperature,
        "sae_type": sae_type,
        "total_time_s": total_time,
        "conditions": condition_means,
        "tests": test_results,
        "per_prompt": per_prompt_details,
    }
    if additive and additive_push_per_head is not None:
        result["additive_push_per_head"] = {
            str(h): {str(f): v for f, v in pushes.items()}
            for h, pushes in additive_push_per_head.items()
        }
        result["additive_boost_strength"] = additive_boost_strength
    # Backward compat: single-head experiments also get "head" and "boundary_features"
    if len(heads_list) == 1:
        result["head"] = heads_list[0]
        result["boundary_features"] = all_boundary_features[heads_list[0]]
    return result

# Print summary

def print_summary(result: dict) -> None:
    """Print a clean summary table of the experiment results."""
    n_prompts = result["n_prompts"]
    n_tokens = result["n_tokens"]
    boost = result["boost_scale"]
    suppress = result["suppress_scale"]
    layer = result["layer"]

    # Multi-head or single-head
    heads = result.get("heads", [result.get("head", "?")])
    n_heads = len(heads)

    if n_heads == 1:
        print(f"\nGeneration Intervention: L{layer}H{heads[0]} boundary features")
        features = result.get("boundary_features", [])
        print(f"Features: {features}")
    else:
        print(f"\nGeneration Intervention: L{layer} x {n_heads} heads {heads}")
        bfph = result.get("boundary_features_per_head", {})
        total_features = sum(len(v) for v in bfph.values())
        print(f"Total features across heads: {total_features}")
        for h_str in sorted(bfph, key=lambda x: int(x)):
            print(f"  H{h_str}: {bfph[h_str]}")

    print(f"Prompts: {n_prompts}, Tokens: {n_tokens}, Boost: {boost}x, Suppress: {suppress}x")
    print(f"Time: {result['total_time_s']:.0f}s")
    print()

    conds = result["conditions"]
    tests = result["tests"]

    header = (
        f"{'Statistic':<22} | {'Baseline':>9} | {'Boost':>9} | {'Suppress':>9} "
        f"| {'Boost p':>9} | {'Supp p':>9} | {'Boost d':>8} | {'Supp d':>8}"
    )
    print(header)
    print("-" * len(header))

    for stat in conds["baseline"]:
        base_val = conds["baseline"][stat]
        boost_val = conds["boost"][stat]
        supp_val = conds["suppress"][stat]
        boost_t = tests["boost_vs_baseline"][stat]
        supp_t = tests["suppress_vs_baseline"][stat]
        print(
            f"{stat:<22} | {base_val:>9.2f} | {boost_val:>9.2f} | {supp_val:>9.2f} "
            f"| {boost_t['p_value']:>9.4f} | {supp_t['p_value']:>9.4f} "
            f"| {boost_t['cohens_d']:>+8.3f} | {supp_t['cohens_d']:>+8.3f}"
        )
