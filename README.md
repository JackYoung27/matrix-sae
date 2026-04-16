# matrix-sae

Sparse autoencoders for matrix-valued recurrent states in GatedDeltaNet (Qwen3.5), with cross-scale (4B) and cross-architecture (DeltaNet 1.3B, GLA 1.3B, Mamba-2) validation.

**The encoder matters more than the decoder.** Across a 720-run swap ablation, a bilinear encoder paired with a flat decoder (`bilinear_flat`) wins on downstream perplexity under state replacement (+0.58%), beating both the full bilinear architecture (+1.33%) and the flat baseline (+3.31%). Flat SAEs win MSE on every layer but lose downstream; the gap traces to residual coherence (flat errors are 5.2x more correlated across heads than bilinear's).

Paper: [paper/matrix_sae.pdf](paper/matrix_sae.pdf). Reproduction steps: [REPRODUCE.md](REPRODUCE.md). Checkpoints: [huggingface.co/JackYoung27/matrix-sae](https://huggingface.co/JackYoung27/matrix-sae).

## Variants

| type | encoder | decoder | downstream PPL (L9, 0.8B) |
| --- | --- | --- | ---: |
| `bilinear_flat` | v_i^T S w_i | flat | +0.58% |
| `bilinear` | v_i^T S w_i | rank-1 | +1.33% |
| `flat` | linear on vec(S) | flat | +3.31% |
| `bilinear_tied` | v_i^T S v_i | rank-1 | +7.05% |
| `rank1` | linear on vec(S) | rank-1 | +11.18% |

The spectral ratio sigma_1/sigma_2 on raw state samples predicts which layers favor rank-1 decoding (rho=0.58 pooled across 21 layers from GDN and DeltaNet).

## Quick start

```bash
pip install -e .
python -m experiments.extraction.extract_states --model Qwen/Qwen3.5-0.8B --layers 9 --n_samples 50000 --output_dir states
python -m core.train --sae_type bilinear --data_dir states --layer 9 --head 0 --n_features 2048 --k 32 --output_dir ckpt
python -m experiments.analysis.analyze --sae_checkpoint ckpt/bilinear_L9_H0/best.pt --data_dir states --layer 9 --head 0 --output_dir out
```

Each script under `experiments/` uses argparse; pass `--help` for flags.

## Layout

```
core/         SAE architectures, training loop, shared types
experiments/  CLI drivers for each paper experiment
paper/        matrix_sae.pdf
pyproject.toml
README.md
REPRODUCE.md
```

## Citation

```bibtex
@unpublished{young2026matrixsae,
  title={Sparse Autoencoders for Matrix-Valued Recurrent States},
  author={Young, Jack},
  note={Manuscript},
  year={2026}
}
```

MIT. See [LICENSE](LICENSE).
