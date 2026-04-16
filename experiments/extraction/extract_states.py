#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

def get_gdn_layer_indices(config) -> list[int]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        # GLA and other pure-linear-attention models have no layer_types attribute;
        # every layer is linear attention.
        return list(range(config.num_hidden_layers))
    return [i for i, t in enumerate(layer_types) if t == "linear_attention"]

def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # HF's modeling_qwen3_5 already uses FLA triton kernels when flash-linear-attention
    # is installed (checked via is_flash_linear_attention_available at import time).
    # No monkey-patching needed.
    config = getattr(model.config, "text_config", model.config)
    gdn_indices = get_gdn_layer_indices(config)
    try:
        if gdn_indices:
            layer0 = (model.model.layers if hasattr(model, "model") else model.layers)[gdn_indices[0]]
            fn = layer0.linear_attn.chunk_gated_delta_rule
            print(f"GDN kernel: {fn.__module__}.{fn.__name__}")
    except AttributeError:
        # Non-Qwen models (GLA, etc.) don't expose chunk_gated_delta_rule
        print(f"Model has {len(gdn_indices)} linear attention layers (no GDN kernel attribute)")

    return model, tokenizer, config

def load_corpus_tokens(
    tokenizer, corpus_path: str | None, seq_len: int, n_samples: int, batch_size: int,
) -> list[torch.Tensor]:
    # tokenize from file or streaming openwebtext, return list of (batch_size, seq_len) tensors
    if corpus_path is not None:
        print(f"Loading corpus from: {corpus_path}")
        token_ids = tokenizer.encode(Path(corpus_path).read_text(), add_special_tokens=False)
    else:
        print("Streaming corpus from Skylion007/openwebtext")
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        token_ids = []
        target = n_samples * seq_len * 2
        for example in ds:
            token_ids.extend(tokenizer.encode(example["text"], add_special_tokens=False))
            if len(token_ids) >= target:
                break
        print(f"Collected {len(token_ids):,} tokens from streaming corpus")

    n_seqs = min(n_samples, len(token_ids) // seq_len)
    if n_seqs < n_samples:
        print(f"Warning: corpus only yields {n_seqs} sequences (requested {n_samples})")

    batches = []
    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    return batches

def save_corpus_tokens(
    tokenizer, corpus_path: str | None, seq_len: int, n_samples: int, output_path: str,
) -> int:
    """Tokenize corpus once and save as a flat .npy file. Returns actual number of sequences."""
    if corpus_path is not None:
        token_ids = tokenizer.encode(Path(corpus_path).read_text(), add_special_tokens=False)
    else:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        token_ids = []
        target = n_samples * seq_len * 2
        for example in ds:
            token_ids.extend(tokenizer.encode(example["text"], add_special_tokens=False))
            if len(token_ids) >= target:
                break
        print(f"Collected {len(token_ids):,} tokens from streaming corpus")

    n_seqs = min(n_samples, len(token_ids) // seq_len)
    # store as (n_seqs, seq_len) int32 array
    arr = np.array(
        [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(n_seqs)],
        dtype=np.int32,
    )
    np.save(output_path, arr)
    print(f"Saved {n_seqs} tokenized sequences to {output_path}")
    return n_seqs

def load_corpus_from_file(corpus_npy_path: str, batch_size: int, n_samples: int | None = None) -> list[torch.Tensor]:
    """Load pre-tokenized corpus .npy and return batches of token tensors."""
    arr = np.load(corpus_npy_path, mmap_mode="r")  # (n_seqs, seq_len)
    if n_samples is not None:
        arr = arr[:n_samples]
    batches = []
    for start in range(0, len(arr), batch_size):
        end = min(start + batch_size, len(arr))
        batches.append(torch.tensor(arr[start:end], dtype=torch.long))
    return batches

def decode_batch_texts(tokenizer, batches: list[torch.Tensor]) -> list[str]:
    """Decode each sequence in the batches back to text strings."""
    texts = []
    for batch in batches:
        for seq in batch:
            texts.append(tokenizer.decode(seq.tolist(), skip_special_tokens=False))
    return texts

def _get_recurrent_state(cache, layer_idx):
    """Extract recurrent state from either Qwen-style or FLA-style cache.

    Qwen: cache.layers[layer_idx].recurrent_states  (attribute, plural)
    FLA:  cache[layer_idx]["recurrent_state"]         (dict, singular)
    """
    # Qwen path: cache.layers[i].recurrent_states
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer_cache = layers[layer_idx]
        state = getattr(layer_cache, "recurrent_states", None)
        if state is not None:
            return state

    # FLA path: cache[i]["recurrent_state"]
    try:
        entry = cache[layer_idx]
        if isinstance(entry, dict) and "recurrent_state" in entry:
            return entry["recurrent_state"]
    except (TypeError, KeyError, IndexError):
        pass

    raise AttributeError(
        f"Cannot extract recurrent state from cache at layer {layer_idx}. "
        f"Cache type: {type(cache).__name__}"
    )

def setup_memmaps(
    output_dir: Path, layer_indices: list[int], n_heads: int,
    key_dim: int, val_dim: int, n_samples: int, resume: bool = False,
) -> dict[int, list[np.memmap]]:
    # one memmap per layer per head: layer_{i}/head_{h}.npy, shape (n_samples, key_dim, val_dim)
    # resume=True opens existing files in r+ mode instead of recreating them
    memmaps: dict[int, list[np.memmap]] = {}
    for layer_idx in layer_indices:
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        heads = []
        for h in range(n_heads):
            fpath = str(layer_dir / f"head_{h}.npy")
            if resume and Path(fpath).exists():
                heads.append(np.lib.format.open_memmap(
                    fpath, mode="r+", dtype=np.float16, shape=(n_samples, key_dim, val_dim),
                ))
            else:
                heads.append(np.lib.format.open_memmap(
                    fpath, mode="w+", dtype=np.float16, shape=(n_samples, key_dim, val_dim),
                ))
        memmaps[layer_idx] = heads
    return memmaps

def probe_state_dims(model, layer_idx: int, tokenizer, device: str) -> tuple[int, int, int]:
    # single forward pass to discover (n_heads, key_dim, val_dim)
    probe_ids = torch.tensor([[tokenizer.eos_token_id] * 16], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=probe_ids, use_cache=True)
    state = _get_recurrent_state(out.past_key_values, layer_idx)
    _, n_heads, key_dim, val_dim = state.shape
    del out
    if device == "cuda":
        torch.cuda.empty_cache()
    return n_heads, key_dim, val_dim

@torch.no_grad()
def extract_states(
    model, config, batches: list[torch.Tensor],
    layer_indices: list[int], memmaps: dict[int, list[np.memmap]], device: str = "cuda",
    start_batch: int = 0, start_offset: int = 0, progress_path: Path | None = None,
) -> int:
    # start_batch / start_offset allow resuming from a partial extraction
    sample_offset = start_offset
    total_samples = sum(batch.shape[0] for batch in batches)
    for batch_idx, batch in enumerate(tqdm(batches, desc="Extracting states")):
        if batch_idx < start_batch:
            continue
        input_ids = batch.to(device)
        bs = input_ids.shape[0]

        outputs = model(input_ids=input_ids, use_cache=True)
        cache = outputs.past_key_values

        for layer_idx in layer_indices:
            # state shape: (batch, num_value_heads, key_head_dim, value_head_dim)
            state_np = _get_recurrent_state(cache, layer_idx).float().cpu().numpy().astype(np.float16)
            for h in range(state_np.shape[1]):
                memmaps[layer_idx][h][sample_offset : sample_offset + bs] = state_np[:, h]

        sample_offset += bs
        del outputs, cache
        if device == "cuda":
            torch.cuda.empty_cache()

        # write progress after each batch so we can resume on crash
        if progress_path is not None:
            for layer_idx in layer_indices:
                for mm in memmaps[layer_idx]:
                    mm.flush()
            progress_path.write_text(json.dumps({
                "batches_done": batch_idx + 1, "samples_written": sample_offset,
                "batch_count": len(batches), "n_samples_total": total_samples,
                "seq_len": int(batch.shape[1]),
            }))

    return sample_offset

def main():
    parser = argparse.ArgumentParser(description="Extract GDN recurrent states for matrix SAE training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--corpus", type=str, default=None, help="Text file path, or streams openwebtext")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 12, 22])
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="./states")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config = load_model_and_tokenizer(args.model, args.device)
    all_gdn_layers = get_gdn_layer_indices(config)
    print(f"GDN layers in model: {all_gdn_layers}")
    print(f"Extracting from layers: {args.layers}")

    invalid = [l for l in args.layers if l not in all_gdn_layers]
    if invalid:
        print(f"Error: layers {invalid} are not GDN layers. Valid: {all_gdn_layers}")
        sys.exit(1)

    n_heads, key_dim, val_dim = probe_state_dims(model, args.layers[0], tokenizer, args.device)
    print(f"State dimensions: {n_heads} heads x ({key_dim}, {val_dim}) per head")
    print(f"Storage per head: {args.n_samples * key_dim * val_dim * 2 / 1e6:.1f} MB")

    batches = load_corpus_tokens(tokenizer, args.corpus, args.seq_len, args.n_samples, args.batch_size)
    actual_samples = sum(b.shape[0] for b in batches)
    if actual_samples == 0:
        print("Error: corpus yielded 0 samples. Check --corpus path or network for streaming.")
        sys.exit(1)
    print(f"Prepared {len(batches)} batches, {actual_samples} samples total")

    memmaps = setup_memmaps(output_dir, args.layers, n_heads, key_dim, val_dim, actual_samples)

    t0 = time.time()
    n_written = extract_states(model, config, batches, args.layers, memmaps, args.device)
    elapsed = time.time() - t0

    for layer_idx in args.layers:
        for mm in memmaps[layer_idx]:
            mm.flush()

    # save texts for interpretability analysis
    texts = decode_batch_texts(tokenizer, batches)[:n_written]
    (output_dir / "texts.json").write_text(json.dumps(texts))
    print(f"Saved {len(texts)} text sequences to texts.json")

    # save metadata
    metadata = {
        "model": args.model,
        "corpus": args.corpus or "Skylion007/openwebtext (streaming)",
        "layer_indices": args.layers,
        "all_gdn_layers": all_gdn_layers,
        "n_samples": n_written,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "state_shape_per_head": [n_written, key_dim, val_dim],
        "dtype": "float16",
        "seq_len": args.seq_len,
        "bytes_per_head": n_written * key_dim * val_dim * 2,
        "extraction_time_s": round(elapsed, 1),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total_bytes = n_written * key_dim * val_dim * 2 * n_heads * len(args.layers)
    print(f"\nDone. {n_written} samples from {len(args.layers)} layers in {elapsed:.1f}s")
    print(f"Output: {output_dir} ({total_bytes / 1e9:.2f} GB)")

if __name__ == "__main__":
    main()
