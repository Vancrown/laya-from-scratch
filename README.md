# Laya from Scratch

An **educational research prototype** exploring bidirectional Transformer decision models, supervised fine-tuning (SFT), probability calibration, RLCD-inspired policy gradients, and continuous-action GRPO-style optimization. **Not an official Laya implementation.**

## Installation

Requires Python 3.12+ and internet access for the initial Hugging Face `bert-base-uncased` download.

```bash
uv sync --frozen
uv run python -c "import laya; print(laya.__file__)"
uv run jupyter lab
```

If the committed lockfile is incompatible with your platform or dependency versions, run `uv lock` and then `uv sync` (record the resulting environment when reporting experiments).

## Layout

- `src/laya/`: decision modeling, SFT, calibration, RLCD, GRPO-style optimization, evaluation, and inference pipeline.
- `notebook/`: numbered instructional notebooks (`01` through `15`; numbering reflects the actual repository).
- `notebook/10_sft.ipynb`: synthetic customer-support SFT example.
- `notebook/14_eval.ipynb`: model evaluation and diagnostics.
- `notebook/15_end_to_end.ipynb`: calibration, inference and checkpoint round-trip.

## Running notebooks

Run the notebooks in numerical order. Later notebooks may depend on checkpoints produced by earlier notebooks; these checkpoints are **not included** in the source archive. The end-to-end notebook expects `../model/grpo_model.pt` relative to `notebook/`; run its training notebook first and ensure that checkpoint exists. Run Jupyter with the repository environment selected as kernel.

The included dataset is a **small synthetic demonstration**, not a validated benchmark. The example calibration set is illustrative only; it must be independent of training and test data for meaningful reported calibration. Do not interpret training curves or tiny-sample accuracy as generalization evidence.

## Model and evaluation

Raw context, question and options are encoded with a pretrained BERT backbone; marker hidden states are scored by a decision head. The pipeline returns option probabilities and a threshold-based or cost-based review/automatic routing decision. The research notebooks explore SFT, REINFORCE-style RLCD, Gaussian-logit GRPO-style training, temperature scaling, NLL, Brier, ECE, realized utility, and option-order sensitivity.

**Limitations:** GRPO here uses continuous Gaussian perturbations of logits, not canonical token-level language-model GRPO. The demo is not production-ready, the illustrative cost matrix is not a real business utility model, and pretrained model downloads plus hardware/software versions can affect reproducibility.

## License

Apache-2.0; see `LICENSE`.
