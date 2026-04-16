
import json
import time
from pathlib import Path

import numpy as np

from core.split_utils import make_train_val_indices, select_sample_indices

def run_baselines(
    states_dir: str,
    layer: int,
    head: int = 0,
    k_values: list[int] | None = None,
    max_samples: int = 10000,
) -> dict[str, object]:
    """Fit PCA and NMF on flattened GDN states, return MSE per k.

    Args:
        states_dir: path to /data/states (contains layer_*/head_*.npy)
        layer: which layer's states to load
        head: which head (default 0)
        k_values: component counts to evaluate (default [1,2,4,8,16,32,64,128])
        max_samples: cap on TRAINING samples used to fit PCA/NMF.
            Validation always uses the full held-out split so MSE stays
            comparable to SAE checkpoints from train.py.

    Returns:
        dict with keys "pca", "nmf" (if applicable), "data_stats".
        Each baseline maps str(k) -> MSE float.
    """
    from sklearn.decomposition import PCA, NMF  # type: ignore[import-untyped]

    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128]

    # Load state data
    layer_dir = Path(states_dir) / f"layer_{layer}"
    data_path = layer_dir / f"head_{head}.npy"
    meta_path = layer_dir / "layer_metadata.json"

    if not data_path.exists():
        raise FileNotFoundError(f"No state data at {data_path}")

    # Read metadata to get correct shape
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_samples = meta["n_samples"]
        d_k = meta["key_head_dim"]
        d_v = meta["value_head_dim"]
    else:
        # fallback: try unified metadata
        unified = Path(states_dir) / "metadata.json"
        if unified.exists():
            meta = json.loads(unified.read_text())
            n_samples = meta["n_samples"]
            d_k = meta["key_head_dim"]
            d_v = meta["value_head_dim"]
        else:
            raise FileNotFoundError(f"No metadata found for layer {layer}")

    print(f"Loading states: layer={layer} head={head} shape=({n_samples}, {d_k}, {d_v})")
    data = np.lib.format.open_memmap(
        str(data_path), mode="r", dtype=np.float16,
        shape=(n_samples, d_k, d_v),
    )

    # Load the full dataset so the held-out split matches train.py exactly.
    raw = data[:].astype(np.float32)

    # flatten to (N, d_k * d_v)
    N = raw.shape[0]
    d_in = d_k * d_v
    X = raw.reshape(N, d_in)

    # data statistics
    data_stats = {
        "n_samples": N,
        "d_k": d_k,
        "d_v": d_v,
        "d_in": d_in,
        "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
        "total_variance": float(X.var()),
        "min_value": float(X.min()),
        "max_value": float(X.max()),
        "fraction_negative": float((X < 0).mean()),
    }
    print(f"  N={N}, d_in={d_in}, variance={data_stats['total_variance']:.6f}, "
          f"min={data_stats['min_value']:.4f}, max={data_stats['max_value']:.4f}, "
          f"frac_negative={data_stats['fraction_negative']:.3f}")

    # Train/val split: match train.py exactly.
    train_idx, val_idx = make_train_val_indices(N, train_fraction=0.8, seed=42)
    X_train_full, X_val = X[train_idx], X[val_idx]

    # Optional cap applies only to the fit set. Keep the full validation split.
    if max_samples and len(X_train_full) > max_samples:
        train_cap_idx = select_sample_indices(len(X_train_full), max_samples, seed=42)
        X_train = X_train_full[train_cap_idx]
        print(f"  Capping PCA/NMF fit set to {len(X_train)} training samples "
              f"(validation still uses {len(X_val)} held-out samples)")
    else:
        X_train = X_train_full

    # Filter k values to be <= d_in and <= n_train
    max_k = min(d_in, len(X_train))
    k_values = [k for k in k_values if k <= max_k]

    print(f"  Fitting PCA (max k={max(k_values)})...")
    t0 = time.time()
    # Fit once at largest k, evaluate at each k by truncating
    max_components = max(k_values)
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(X_train)
    pca_time = time.time() - t0
    print(f"  PCA fit in {pca_time:.1f}s")

    pca_results = {}
    for k in k_values:
        # project to k components and reconstruct
        components = pca.components_[:k]  # (k, d_in)
        mean = pca.mean_  # (d_in,)
        X_centered = X_val - mean
        projected = X_centered @ components.T  # (N_val, k)
        reconstructed = projected @ components + mean  # (N_val, d_in)
        mse = float(np.mean((X_val - reconstructed) ** 2))
        pca_results[str(k)] = mse
        ev = 1.0 - mse / max(X_val.var(), 1e-12)
        print(f"    PCA-{k}: MSE={mse:.6f} EV={ev:.4f}")

    pca_results["fit_time_s"] = round(pca_time, 2)

    nmf_results: dict[str, float | None] = {}
    if data_stats["fraction_negative"] < 0.01:
        # data is non-negative enough for NMF
        print(f"  Fitting NMF (data is {100*(1-data_stats['fraction_negative']):.1f}% non-negative)")
        X_train_nn = np.maximum(X_train, 0)
        X_val_nn = np.maximum(X_val, 0)

        for k in k_values:
            t0 = time.time()
            try:
                nmf = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)  # pyright: ignore[reportArgumentType]
                nmf.fit_transform(X_train_nn)
                H = nmf.components_  # (k, d_in)
                W_val = nmf.transform(X_val_nn)
                reconstructed = W_val @ H
                # MSE against original (non-clipped) validation data
                mse = float(np.mean((X_val - reconstructed) ** 2))
                nmf_results[str(k)] = mse
                ev = 1.0 - mse / max(X_val.var(), 1e-12)
                elapsed = time.time() - t0
                print(f"    NMF-{k}: MSE={mse:.6f} EV={ev:.4f} [{elapsed:.1f}s]")
            except Exception as e:
                print(f"    NMF-{k}: FAILED ({e})")
                nmf_results[str(k)] = None
    else:
        # data has substantial negative values, apply ReLU and warn
        print(f"  Data is {100*data_stats['fraction_negative']:.1f}% negative. "
              f"Fitting NMF on max(0, x).")
        X_train_nn = np.maximum(X_train, 0)
        X_val_nn = np.maximum(X_val, 0)

        for k in k_values:
            t0 = time.time()
            try:
                nmf = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)  # pyright: ignore[reportArgumentType]
                nmf.fit_transform(X_train_nn)
                H = nmf.components_
                W_val = nmf.transform(X_val_nn)
                reconstructed = W_val @ H
                # MSE against original validation data (including negatives NMF can't represent)
                mse = float(np.mean((X_val - reconstructed) ** 2))
                nmf_results[str(k)] = mse
                ev = 1.0 - mse / max(X_val.var(), 1e-12)
                elapsed = time.time() - t0
                print(f"    NMF-{k}: MSE={mse:.6f} EV={ev:.4f} [{elapsed:.1f}s]")
            except Exception as e:
                print(f"    NMF-{k}: FAILED ({e})")
                nmf_results[str(k)] = None

    return {
        "pca": pca_results,
        "nmf": nmf_results,
        "data_stats": data_stats,
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PCA/NMF baselines for GDN states")
    p.add_argument("--data-dir", required=True, help="path to states directory")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--output", default=None, help="output JSON path")
    args = p.parse_args()

    result = run_baselines(
        states_dir=args.data_dir,
        layer=args.layer,
        head=args.head,
        max_samples=args.max_samples,
    )

    output = args.output or f"baselines_layer{args.layer}.json"
    Path(output).write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output}")
