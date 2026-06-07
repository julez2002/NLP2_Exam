# NLP2 Exam: Code Overview

Fine-tunes **ModernBERT-base** (`answerdotai/ModernBERT-base`) on a multi-genre dataset to classify texts as human-written or AI-generated. Two model variants (+ baseline) are compared, each available in two auxiliary-loss weight configurations (`0.1` and `0.5`).

---

## File Overview

### Training scripts (`Ucloud_*.py`)

| File | Model | Description |
|------|-------|-------------|
| `Ucloud_baseline.py` | **Baseline** | Single binary head: human vs. AI |
| `Ucloud_6head_final.py` | **6-Way Head** | Binary head + 6-way auxiliary head (genre × source interaction) |
| `Ucloud_3genre_final.py` | **3-Way Head** | Binary head + 3-way auxiliary head (genre classification) |

The `_0.5` variants repeat training with auxiliary loss weight `λ = 0.5` instead of `0.1`.

**Architecture (shared):**
- Encoder: ModernBERT-base (768-dim hidden, up to 4096 tokens)
- Pooling: masked mean pooling over all token representations
- Dropout: 0.1
- Loss: inverse-frequency weighted cross-entropy; auxiliary head weighted by `λ` (GENRE_WEIGHT)

**Training setup:**
- AdamW, lr = 2e-5, weight decay = 0.01
- Batch size 16 (physical), gradient accumulation ×4 → effective batch 64
- Linear warmup (10%) + linear decay
- Early stopping: patience 4, min delta 5e-3 on macro-F1
- 3 seeds (42, 123, 7); best checkpoint per seed saved

**Data format** (`merged_train.jsonl` / `val.jsonl`): one JSON object per line with fields `text`, `label` (0 = human, 1 = AI), `genre` (fiction / news / essays), and optionally `attack` (obfuscation type).

---

### Validation scripts (`val_*.py`)

Run against **`val.jsonl`** using saved checkpoints from training.

| File | Matches training script |
|------|------------------------|
| `val_baseline.py` | `Ucloud_baseline.py` |
| `val_6head.py` | `Ucloud_6head_final.py` (λ = 0.1) |
| `val_3genre_0.1.py` | `Ucloud_3genre_final.py` (λ = 0.1) |

Each val script loads the three seed checkpoints, evaluates each on the validation set, then prints and saves aggregate metrics (mean ± std).

**Metrics reported:**
- Binary F1, macro F1, AUC-ROC, accuracy
- TPR at 5% FPR
- PAN metrics: Brier score, c@1, F0.5u
- Source silhouette score (cosine, pooled embeddings)
- Per-genre binary F1 (fiction / news / essays)
- Clean vs. attacked breakdown (binary F1, AUC-ROC, TPR@5FPR)

Results saved to `<SAVE_PATH>_<label>_val_results.json`.

---

### Test / final evaluation scripts (`evaluate_*.py`)

Identical to the val scripts but run against **`test.jsonl`**. Additionally report per-prompt-type accuracy and mean AI probability (`ai_clean`, `ai_obf1`–`ai_obf4`, `human`).

| File | Model variant |
|------|--------------|
| `evaluate_baseline.py` | Baseline |
| `evaluate_6head.py` | 6-Way Head (λ = 0.1) |
| `evaluate_3genre.py` | 3-Way Head (λ = 0.1) |

Results saved to `<SAVE_PATH>_<label>_test_results.json`.

> Scripts were developed for UCloud (NVIDIA B200, 192 GB VRAM) with `MAX_LEN = 4096`. On smaller GPUs reduce `MAX_LEN` and `BATCH_SIZE` accordingly.

`torch` (pre-installed on UCloud), `numpy`, `transformers`, `accelerate`, `scikit-learn`

