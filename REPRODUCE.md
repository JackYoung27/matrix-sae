# Reproducing the paper

Each experiment runs on one GPU via argparse. Run from the repo root so `core.*` and `experiments.*` resolve.

## 1. Install

```
git clone <this-repo> matrix-sae && cd matrix-sae
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`fla` (flash-linear-attention) is needed for DeltaNet and GLA. `mamba-ssm` plus `causal-conv1d` is needed for the Mamba-2 audit. The wheel URLs in git history document the exact versions: installing fresh on torch 2.10 triggers an ABI break in `causal-conv1d`, so pin to torch 2.8 with the corresponding wheels, or rebuild from source.

```
python -m experiments.<name> --help
```

## 2. Hardware

| experiment | script | GPU | time |
| --- | --- | --- | --- |
| Extract 0.8B GDN states, one layer, 50K sequences | `experiments.extraction.extract_states` | 1x A10G or 3090 | 15 min |
| Train one SAE, 0.8B, one head | `core.train` | 1x A10G | 10 min |
| 720-run encoder-swap sweep | `experiments.ablations.encoder_swap_ablation` | 16x A100 parallel | 2 to 3 h (days sequential) |
| Per-layer five-variant sweep (L1, L9, L17) | `experiments.ablations.layer_encoder_swap_ablation` | 8x A100 parallel | 1 to 2 h |
| 4B cross-scale | `experiments.run_9pager_overnight` | 1x A100 80GB | ~1 h |
| DeltaNet and GLA validation | `experiments.run_{deltanet,gla}_validation` | 1x A100 | ~30 min each |
| Mamba-2 spectral audit (no training) | `experiments.mamba2.mamba2_write_geometry` | 1x A100 | ~20 min |
| Downstream eval, one setting | `experiments.run_batchtopk_downstream`, etc. | 1x A10G | 15 to 30 min |

Sweeps run sequentially by default. Parallelize by launching the script multiple times with different `--seed`, `--head`, or `--layer`, or wrap with joblib or SLURM.

## 3. Data

Corpus: OpenWebText for main training, UltraChat 50K slice for cross-corpus checks. Both load via `datasets`.

Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`, `fla-hub/delta_net-1.3B-100B`, `fla-hub/gla-1.3B-100B`, `state-spaces/mamba-2.8b`.

SAE checkpoints: 25-checkpoint inspection pack available on HuggingFace (release pending). Alternative: retrain with `core.train`.

## 4. Reproduction levels

**L1 inspect.** Open `paper/matrix_sae.pdf`, read `core/sae.py`. No GPU.

**L2 load a published SAE.** `torch.load("best.pt", weights_only=False, map_location="cpu")`. 5 min, CPU.

**L3 rerun one downstream eval.** Extract 500 sequences of states, load the matching SAE pack from the HuggingFace release, run `experiments.run_batchtopk_downstream`. 1 h on one A10G.

**L4 retrain one SAE.** `python -m core.train --sae_type bilinear --data_dir states --layer 9 --head 0 --n_features 2048 --k 32 --output_dir ckpt`. 10 min on one A10G.

**L5 paper replication.** Run every row in §2. 100 to 300 GPU-hours on your own parallel infrastructure.

## 5. Expected outputs

Each script writes a JSON under `--output-dir`. JSONs match those cited in the paper. Last-digit variation is expected from nondeterministic tensor ops. Hypothesis-test conclusions (bilinear < flat downstream, sigma_1/sigma_2 predicts decoder choice) are stable across seeds.

## 6. Loading an HF checkpoint

```python
import torch
ckpt = torch.load("bilinear_L9_H0_nf2048_k32_s42/best.pt", weights_only=False, map_location="cpu")
print(ckpt["config"])
print(ckpt["val_mse"])
```

The `manifest.json` in the HF repo maps tags to SHA256 and metadata.

## 7. Known pitfalls

SAE checkpoints save with `torch.save` using a full module pickle. Load with `weights_only=False`.

Older rank-1 checkpoints have 2D decoder atoms `(d_k, d_v)` rather than 3D `(1, d_k, d_v)`. `core.sae.load_sae_checkpoint` upgrades these on load.

State `.npy` files are memory-mapped; put them on SSD.

The 4B model in fp16 uses ~10 GB VRAM for the transformer alone, before SAE replacement.
