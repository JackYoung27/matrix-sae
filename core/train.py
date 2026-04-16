#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sae import FlatSAE, MatrixSAE, BilinearMatrixSAE, BilinearEncoderFlatSAE  # noqa: E402
from core.split_utils import make_train_val_subsets  # noqa: E402

# Packaged layout and Modal flat layout: try package path first, then define
# locally when ``core/types.py`` isn't mounted.
try:
    from core.types import SAEConfig, TrainResult  # noqa: E402
except ImportError:
    from typing import TypedDict

    class SAEConfig(TypedDict, total=False):  # type: ignore[no-redef]
        sae_type: str
        layer: int
        head: int
        n_features: int
        d_k: int
        d_v: int
        d_in: int
        expansion_factor: int
        k: int
        rank: int
        use_batchtopk: bool
        seed: int
        lr: float
        lr_min: float
        batch_size: int
        epochs: int
        total_steps: int
        n_params: int
        device: str
        code_sha: str

    class TrainResult(TypedDict):  # type: ignore[no-redef]
        sae_type: str
        layer: int
        head: int
        expansion_factor: int
        k: int
        rank: int
        seed: int
        code_sha: str
        n_features: int
        n_samples: int
        best_mse: float
        final_mse: float
        final_n_dead: int
        total_time_s: float

TrainableSAE = FlatSAE | MatrixSAE | BilinearMatrixSAE | BilinearEncoderFlatSAE
_wandb = None  # lazy-loaded

class GDNStateDataset(Dataset):
    def __init__(self, path: str):
        self.data = np.load(path, mmap_mode="r")
    def __len__(self) -> int:
        return self.data.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].astype(np.float32))

def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

def _log(log_file, record: dict, step: int) -> None:
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()
    if _wandb is not None:
        _wandb.log(record, step=step)

@torch.no_grad()
def evaluate(model: TrainableSAE, loader: DataLoader, device: torch.device, is_flat: bool) -> dict[str, float]:
    was_training = model.training
    model.eval()
    saved_dead_state = model.steps_since_active.clone() if hasattr(model, "steps_since_active") else None
    totals = {"mse": 0.0, "l0": 0.0}
    feat_ever_active: torch.Tensor | None = None
    n = 0
    for batch in loader:
        batch = batch.to(device)
        bs = batch.shape[0]
        out = model(batch.reshape(bs, -1) if is_flat else batch)
        x_flat, recon_flat = batch.reshape(bs, -1), out.reconstruction.reshape(bs, -1)
        totals["mse"] += F.mse_loss(recon_flat, x_flat).item() * bs
        totals["l0"] += (out.coefficients != 0).float().sum().item()
        batch_active = (out.coefficients.abs() > 0).any(dim=0)
        if feat_ever_active is None:
            feat_ever_active = batch_active
        else:
            feat_ever_active = feat_ever_active | batch_active
        n += bs
    if saved_dead_state is not None:
        model.steps_since_active.copy_(saved_dead_state)
    model.train(was_training)
    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg["dead"] = float((~feat_ever_active).sum().item()) if feat_ever_active is not None else 0.0
    return avg

def _save(model: TrainableSAE, path: str, config: SAEConfig | None = None, **extra: object) -> None:
    payload: dict[str, object] = dict(model_state_dict=model.state_dict(), **extra)
    if config is not None:
        payload["config"] = config
    torch.save(payload, path)

def _clear_optimizer_state_slice(
    optimizer: torch.optim.Optimizer, param: torch.Tensor,
    indices: torch.Tensor, dim: int = 0,
) -> None:
    state = optimizer.state.get(param)
    if not state:
        return
    idx = indices.to(device=param.device, dtype=torch.long)
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(key)
        if not torch.is_tensor(buf) or buf.ndim == 0 or buf.shape != param.shape:
            continue
        buf.index_fill_(dim, idx, 0)

def _clear_resampled_optimizer_state(
    model: TrainableSAE, optimizer: torch.optim.Optimizer, indices: torch.Tensor,
) -> None:
    if indices.numel() == 0:
        return
    if isinstance(model, FlatSAE):
        _clear_optimizer_state_slice(optimizer, model.encoder.weight, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.encoder.bias, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.decoder.weight, indices, dim=1)
    elif isinstance(model, MatrixSAE):
        _clear_optimizer_state_slice(optimizer, model.encoder.weight, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.encoder.bias, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.V, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W, indices, dim=0)
    elif isinstance(model, BilinearEncoderFlatSAE):
        _clear_optimizer_state_slice(optimizer, model.decoder.weight, indices, dim=1)
        _clear_optimizer_state_slice(optimizer, model.b_enc, indices, dim=0)
        if model.V_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
        if model.W_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)
    elif isinstance(model, BilinearMatrixSAE):
        _clear_optimizer_state_slice(optimizer, model.V_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.b_enc, indices, dim=0)
        if not model.tied and model.V_enc is not None and model.W_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
            _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)

def train(
    sae_type: str,
    data_dir: str,
    layer: int,
    head: int,
    n_features: int,
    k: int = 32,
    lr: float = 3e-4,
    lr_min: float = 3e-5,
    batch_size: int = 256,
    epochs: int = 20,
    warmup_steps: int = 50,
    norm_every: int = 100,
    resample_every: int = 250,
    log_every: int = 50,
    output_dir: str = "checkpoints",
    use_wandb: bool = False,
    seed: int = 42,
    rank: int = 1,
    use_batchtopk: bool = False,
) -> TrainResult:
    global _wandb
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(data_dir, f"layer_{layer}", f"head_{head}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"State data not found: {data_path}")
    dataset = GDNStateDataset(data_path)
    if len(dataset) == 0:
        raise ValueError(f"Empty dataset at {data_path}")
    d_k, d_v = dataset.data.shape[1], dataset.data.shape[2]
    d_in = d_k * d_v
    expansion = n_features // d_in if n_features >= d_in else 0

    train_set, val_set = make_train_val_subsets(dataset, train_fraction=0.8, seed=42)
    n_train = len(train_set)

    def make_loader(dataset_split: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    train_loader = make_loader(train_set, True)
    val_loader = make_loader(val_set, False)

    is_flat = sae_type == "flat"
    if is_flat:
        model: TrainableSAE = FlatSAE(d_in=d_in, n_features=n_features, k=k, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear":
        model = BilinearMatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear_tied":
        model = BilinearMatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, tied=True, rank=rank, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear_flat":
        model = BilinearEncoderFlatSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    else:
        model = MatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = epochs * len(train_loader)
    warmup_steps = min(warmup_steps, total_steps // 4)
    resample_every = min(resample_every, max(total_steps // 4, 1))
    print(f"{sae_type} | features={n_features} k={k} params={n_params:,} | "
          f"train={n_train} val={len(dataset)-n_train} | {device}")
    print(f"  total_steps={total_steps} warmup={warmup_steps} resample_every={resample_every}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
        _sha = result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        _sha = ""
    _sha = _sha or os.environ.get("MATRIX_SAE_CODE_SHA", "unknown")
    config: SAEConfig = dict(sae_type=sae_type, layer=layer, head=head, n_features=n_features,
                  d_k=d_k, d_v=d_v, d_in=d_k * d_v, expansion_factor=expansion,
                  k=k, rank=rank, use_batchtopk=use_batchtopk,
                  seed=seed, lr=lr, lr_min=lr_min, batch_size=batch_size,
                  epochs=epochs, total_steps=total_steps, n_params=n_params,
                  device=str(device), code_sha=_sha)  # type: ignore[typeddict-item]
    json.dump(config, open(os.path.join(output_dir, "config.json"), "w"), indent=2)
    log_file = open(os.path.join(output_dir, "train_log.jsonl"), "a")

    if use_wandb:
        import wandb
        _wandb = wandb
        _wandb.init(project="matrix-sae", config=dict(config), name=f"{sae_type}_L{layer}_H{head}")

    best_val_mse, step = float("inf"), 0
    start_epoch = 0
    t_start = time.time()

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resuming from step {step} / epoch {start_epoch} (best_val_mse={best_val_mse:.4e})")

    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_t in pbar:
            batch_t = batch_t.to(device, non_blocking=True)
            bs = batch_t.shape[0]
            out = model(batch_t.reshape(bs, -1) if is_flat else batch_t)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()

            lr_now = get_lr(step, warmup_steps, total_steps, lr, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.step()

            if step % norm_every == 0:
                model.normalize_decoder()
            if step > 0 and step % resample_every == 0:
                resampled = model.resample_dead_features(batch_t)
                if resampled.numel() > 0:
                    _clear_resampled_optimizer_state(model, optimizer, resampled)
                    print(f"  Resampled {int(resampled.numel())} dead features at step {step}")

            if step % log_every == 0:
                with torch.no_grad():
                    x_flat = batch_t.reshape(bs, -1)
                    r_flat = out.reconstruction.reshape(bs, -1)
                    mse = F.mse_loss(r_flat, x_flat).item()
                    nmse = mse / (x_flat ** 2).mean().clamp(min=1e-12).item()
                    ev = 1.0 - mse / x_flat.var().clamp(min=1e-12).item()
                    l0 = (out.coefficients != 0).float().sum(dim=-1).mean().item()
                m = dict(step=step, epoch=epoch, lr=lr_now, loss=out.loss.item(),
                         mse=mse, nmse=nmse, explained_var=ev, l0=l0)
                pbar.set_postfix(loss=f"{m['loss']:.4e}", nmse=f"{nmse:.4f}",
                                 ev=f"{ev:.4f}", l0=f"{l0:.0f}", lr=f"{lr_now:.2e}")
                _log(log_file, m, step)
            step += 1

        val = evaluate(model, val_loader, device, is_flat)
        vr = {f"val_{vk}": vv for vk, vv in val.items()}
        vr.update(epoch=epoch, step=step)
        _log(log_file, vr, step)
        print(f"  Val MSE={val['mse']:.4e}  L0={val['l0']:.1f}  Dead={val['dead']:.0f}")

        if val["mse"] < best_val_mse:
            best_val_mse = val["mse"]
            _save(model, os.path.join(output_dir, "best.pt"),
                  epoch=epoch, step=step, val_mse=best_val_mse, config=config)
            print(f"  Saved best (val_mse={best_val_mse:.4e})")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step, "epoch": epoch,
            "best_val_mse": best_val_mse,
            "config": config,
        }, ckpt_path)

    final_val_mse = val.get("mse", best_val_mse) if (epochs > 0 and start_epoch < epochs) else best_val_mse  # type: ignore[possibly-unbound]
    _save(model, os.path.join(output_dir, "final.pt"),
          epoch=epochs - 1, step=step, val_mse=final_val_mse, config=config)
    log_file.close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # clean up resume checkpoint after successful completion
    if _wandb is not None:
        _wandb.finish()
        _wandb = None
    print(f"Saved final model to {output_dir}/final.pt")

    model_dead = int((model.steps_since_active >= model.dead_threshold).sum().item()) if hasattr(model, "steps_since_active") else 0
    return {
        "sae_type": sae_type, "layer": layer, "head": head,
        "expansion_factor": expansion, "k": k, "rank": rank, "seed": seed,
        "code_sha": _sha, "n_features": n_features,
        "n_samples": len(dataset), "best_mse": best_val_mse,
        "final_mse": final_val_mse, "final_n_dead": model_dead,
        "total_time_s": round(time.time() - t_start, 1),
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_type", choices=["flat", "rank1", "bilinear", "bilinear_tied", "bilinear_flat"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", required=True, help="int or 'all'")
    p.add_argument("--n_features", default="16384", help="int or expansion like '2x'")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=0)  # 0 = let adaptive logic decide
    p.add_argument("--norm_every", type=int, default=100)
    p.add_argument("--resample_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--batchtopk", action="store_true", help="Use BatchTopK sparsity instead of per-sample TopK")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    input_dim = 128 * 128
    nf = args.n_features
    n_features = int(input_dim * float(nf[:-1])) if nf.endswith("x") else int(nf)

    if args.head == "all":
        ld = os.path.join(args.data_dir, f"layer_{args.layer}")
        heads = sorted(int(f.stem.split("_")[1]) for f in Path(ld).glob("head_*.npy"))
        print(f"Training {len(heads)} heads: {heads}")
    else:
        heads = [int(args.head)]

    for head in heads:
        print(f"\n{'='*60}\nlayer={args.layer} head={head} type={args.sae_type}\n{'='*60}")
        train(
            sae_type=args.sae_type, data_dir=args.data_dir, layer=args.layer,
            head=head, n_features=n_features, k=args.k, lr=args.lr,
            lr_min=args.lr_min, batch_size=args.batch_size, epochs=args.epochs,
            warmup_steps=args.warmup_steps, norm_every=args.norm_every,
            resample_every=args.resample_every, log_every=args.log_every,
            output_dir=os.path.join(args.output_dir, f"{args.sae_type}_L{args.layer}_H{head}"),
            use_wandb=args.wandb, rank=args.rank,
            use_batchtopk=args.batchtopk,
        )

if __name__ == "__main__":
    main()
