import subprocess
subprocess.run(
    ["pip", "install", "-q", "numpy", "transformers", "accelerate", "scikit-learn"],
    check=True,
)

# ── 1. Imports ─────────────────────────────────────────────────────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, silhouette_score
from tqdm.auto import tqdm

# ── 2. Config — must match training script exactly ─────────────────────────────
MODEL_NAME   = "answerdotai/ModernBERT-base"
TEST_PATH    = "val.jsonl"          # ← set to your test file
SAVE_PATH    = "3genrehead_0.1"
MAX_LEN      = 4096
BATCH_SIZE   = 16
GENRE_WEIGHT = 0.1
SEEDS        = [42, 123, 7]

GENRE_MAP = {"fiction": 0, "news": 1, "essays": 2}

PROMPT_LABELS = {
    0:  "ai_clean",
    1:  "ai_obf1",
    2:  "ai_obf2",
    3:  "ai_obf3",
    4:  "ai_obf4",
    99: "human",
}

_cache_tag  = f"{MODEL_NAME.replace('/', '_')}_{MAX_LEN}"
TEST_CACHE  = f"val_tokenized_{_cache_tag}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 3. Data utilities ──────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# ── 4. Dataset ─────────────────────────────────────────────────────────────────
class AIDetectionDataset(Dataset):
    def __init__(self, records: list, tokenizer) -> None:
        self.data = []
        for r in tqdm(records, desc="Tokenising", leave=False):
            enc = tokenizer(
                r["text"],
                truncation=True,
                max_length=MAX_LEN,
                add_special_tokens=True,
            )
            self.data.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "source_label":   r.get("label", -1),
                "genre_label":    GENRE_MAP.get(r.get("genre", ""), -1),
                "attack":         r.get("attack"),
                "prompt":         int(r.get("prompt", -1)),
            })

    def save(self, path: str) -> None:
        torch.save(self.data, path)

    @classmethod
    def from_cache(cls, path: str) -> "AIDetectionDataset":
        obj      = cls.__new__(cls)
        obj.data = torch.load(path, weights_only=False)
        return obj

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

# ── 5. Collate ─────────────────────────────────────────────────────────────────
def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask = [], []
        source_labels, genre_labels, attacks, prompts = [], [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"]       + [pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            source_labels.append(x["source_label"])
            genre_labels.append(x["genre_label"])
            attacks.append(x["attack"])
            prompts.append(x.get("prompt", -1))
        return {
            "input_ids":      torch.tensor(input_ids,     dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "source_label":   torch.tensor(source_labels,  dtype=torch.long),
            "genre_label":    torch.tensor(genre_labels,   dtype=torch.long),
            "attack":         attacks,
            "prompt":         prompts,
        }
    return collate_fn

# ── 6. Model ───────────────────────────────────────────────────────────────────
class MultiTaskModernBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout     = nn.Dropout(0.1)
        self.source_head = nn.Linear(hidden, 2)
        self.genre_head  = nn.Linear(hidden, 3)

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, return_embeddings: bool = False):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.mean_pool(out.last_hidden_state, attention_mask))
        if return_embeddings:
            return self.source_head(pooled), self.genre_head(pooled), pooled
        return self.source_head(pooled), self.genre_head(pooled)

# ── 7. PAN evaluation metrics ─────────────────────────────────────────────────
def pan_brier(probs: list, labels: list) -> float:
    """PAN Brier score = 1 - mean squared error (higher is better)."""
    p = np.array(probs, dtype=float)
    y = np.array(labels, dtype=float)
    return float(1.0 - np.mean((p - y) ** 2))


def pan_c_at_1(probs: list, labels: list) -> float:
    """c@1: accuracy that rewards abstaining on uncertain cases (score==0.5)."""
    p = np.array(probs, dtype=float)
    y = np.array(labels, dtype=float)
    n  = len(p)
    nu = int(np.sum(p == 0.5))
    preds = np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, -1.0))
    nc = int(np.sum((preds >= 0) & (preds == y)))
    return float((nc + nu * nc / n) / n)


def pan_f05u(probs: list, labels: list) -> float:
    """F0.5u: precision-weighted F-score; unanswered (score==0.5) count as half-wrong."""
    p = np.array(probs, dtype=float)
    y = np.array(labels, dtype=float)
    n_u  = int(np.sum(p == 0.5))
    n_tp = int(np.sum((p > 0.5) & (y == 1)))
    n_fp = int(np.sum((p > 0.5) & (y == 0)))
    n_fn = int(np.sum((p < 0.5) & (y == 1)))
    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp
    return float(1.25 * n_tp / denom) if denom > 0 else 0.0


# ── 9. Evaluation — identical to training script ───────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()

    all_probs, all_preds, all_labels  = [], [], []
    all_genre_preds, all_genre_labels = [], []
    all_attacks:    list = []
    all_prompts:    list = []
    all_embeddings: list = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            source_logits, genre_logits, pooled = model(ids, mask, return_embeddings=True)

        all_probs.extend(F.softmax(source_logits, dim=-1)[:, 1].cpu().tolist())
        all_preds.extend(source_logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["source_label"].tolist())
        all_genre_preds.extend(genre_logits.argmax(-1).cpu().tolist())
        all_genre_labels.extend(batch["genre_label"].tolist())
        all_attacks.extend(batch["attack"])
        all_prompts.extend(batch["prompt"])
        all_embeddings.append(pooled.float().cpu())

    _fprs, _tprs, _threshs = roc_curve(all_labels, all_probs)
    _valid     = np.where(_fprs <= 0.05)[0]
    _t5_thresh = float(_threshs[_valid[-1]]) if len(_valid) else 1.0

    _has_genre = any(g >= 0 for g in all_genre_labels)

    metrics = {
        "binary_f1":   f1_score(all_labels, all_preds, average="binary", pos_label=1),
        "macro_f1":    f1_score(all_labels, all_preds, average="macro"),
        "auc_roc":     roc_auc_score(all_labels, all_probs),
        "accuracy":    accuracy_score(all_labels, all_preds),
        "tpr_at_5fpr": float(_tprs[_valid[-1]]) if len(_valid) else 0.0,
        "brier":       pan_brier(all_probs, all_labels),
        "c_at_1":      pan_c_at_1(all_probs, all_labels),
        "f05u":        pan_f05u(all_probs, all_labels),
    }
    if _has_genre:
        metrics["genre_macro_f1"] = f1_score(all_genre_labels, all_genre_preds, average="macro")

    _emb   = torch.cat(all_embeddings, 0).numpy()
    _sil_n = min(len(all_labels), 2000)
    metrics["source_silhouette"] = float(silhouette_score(
        _emb, all_labels, metric="cosine", sample_size=_sil_n, random_state=0,
    ))
    if _has_genre:
        metrics["genre_silhouette"] = float(silhouette_score(
            _emb, all_genre_labels, metric="cosine", sample_size=_sil_n, random_state=0,
        ))

    attacked_idx = [i for i, a in enumerate(all_attacks) if a is not None]
    clean_idx    = [i for i, a in enumerate(all_attacks) if a is None]

    if clean_idx:
        c_labels = [all_labels[i] for i in clean_idx]
        c_preds  = [all_preds[i]  for i in clean_idx]
        c_probs  = [all_probs[i]  for i in clean_idx]
        metrics["clean_binary_f1"] = f1_score(c_labels, c_preds, average="binary", pos_label=1)
        if len(set(c_labels)) > 1:
            metrics["clean_auc_roc"] = roc_auc_score(c_labels, c_probs)
            _c_fprs, _c_tprs, _ = roc_curve(c_labels, c_probs)
            _cv = np.where(_c_fprs <= 0.05)[0]
            metrics["clean_tpr_at_5fpr"] = float(_c_tprs[_cv[-1]]) if len(_cv) else 0.0

    if attacked_idx:
        a_labels = [all_labels[i] for i in attacked_idx]
        a_preds  = [all_preds[i]  for i in attacked_idx]
        a_probs  = [all_probs[i]  for i in attacked_idx]
        metrics["attacked_binary_f1"] = f1_score(a_labels, a_preds, average="binary", pos_label=1)
        if len(set(a_labels)) > 1:
            metrics["attacked_auc_roc"] = roc_auc_score(a_labels, a_probs)
            _a_fprs, _a_tprs, _ = roc_curve(a_labels, a_probs)
            _av = np.where(_a_fprs <= 0.05)[0]
            metrics["attacked_tpr_at_5fpr"] = float(_a_tprs[_av[-1]]) if len(_av) else 0.0
        else:
            metrics["attacked_tpr_at_5fpr"] = float(
                np.mean([p >= _t5_thresh for p in a_probs])
            )

    genre_names = {v: k for k, v in GENRE_MAP.items()}
    for g_idx, g_name in genre_names.items():
        g_idxs = [i for i, g in enumerate(all_genre_labels) if g == g_idx]
        if g_idxs:
            g_labels = [all_labels[i] for i in g_idxs]
            g_preds  = [all_preds[i]  for i in g_idxs]
            if len(set(g_labels)) > 1:
                metrics[f"{g_name}_binary_f1"] = f1_score(
                    g_labels, g_preds, average="binary", pos_label=1,
                )

    for p_val, p_name in sorted(PROMPT_LABELS.items()):
        p_idxs = [i for i, p in enumerate(all_prompts) if p == p_val]
        if p_idxs:
            p_labels = [all_labels[i] for i in p_idxs]
            p_preds  = [all_preds[i]  for i in p_idxs]
            p_probs  = [all_probs[i]  for i in p_idxs]
            metrics[f"{p_name}_acc"]       = accuracy_score(p_labels, p_preds)
            metrics[f"{p_name}_mean_prob"] = float(np.mean(p_probs))

    return metrics

# ── 10. Pretty-print helper ────────────────────────────────────────────────────
def print_metrics(metrics: dict, prefix: str = "") -> None:
    line = (
        f"{prefix}"
        f"bin_f1={metrics['binary_f1']:.4f}"
        f"  |  macro_f1={metrics['macro_f1']:.4f}"
        f"  |  auc_roc={metrics['auc_roc']:.4f}"
        f"  |  acc={metrics['accuracy']:.4f}"
        f"  |  tpr@5fpr={metrics['tpr_at_5fpr']:.4f}"
        f"  |  src_sil={metrics['source_silhouette']:.4f}"
    )
    if "genre_macro_f1" in metrics:
        line += f"  |  genre_f1={metrics['genre_macro_f1']:.4f}"
    if "genre_silhouette" in metrics:
        line += f"  |  gen_sil={metrics['genre_silhouette']:.4f}"
    print(line)
    pan_line = (
        f"{prefix}"
        f"[PAN] brier={metrics['brier']:.4f}"
        f"  |  c@1={metrics['c_at_1']:.4f}"
        f"  |  F0.5u={metrics['f05u']:.4f}"
    )
    print(pan_line)
    if "clean_binary_f1" in metrics:
        clean_line = f"  clean_f1={metrics['clean_binary_f1']:.4f}"
        if "clean_auc_roc"     in metrics: clean_line += f"  |  clean_auc={metrics['clean_auc_roc']:.4f}"
        if "clean_tpr_at_5fpr" in metrics: clean_line += f"  |  clean_tpr@5fpr={metrics['clean_tpr_at_5fpr']:.4f}"
        print(clean_line)
    if "attacked_binary_f1" in metrics:
        atk_line = f"  attacked_f1={metrics['attacked_binary_f1']:.4f}"
        if "attacked_auc_roc"     in metrics: atk_line += f"  |  attacked_auc={metrics['attacked_auc_roc']:.4f}"
        if "attacked_tpr_at_5fpr" in metrics: atk_line += f"  |  attacked_tpr@5fpr={metrics['attacked_tpr_at_5fpr']:.4f}"
        print(atk_line)
    genre_f1_parts = "  |  ".join(
        f"{g}_f1={metrics[f'{g}_binary_f1']:.4f}"
        for g in ("fiction", "news", "essays")
        if f"{g}_binary_f1" in metrics
    )
    if genre_f1_parts:
        print(f"  per-genre binary: {genre_f1_parts}")
    prompt_order = ["ai_clean", "ai_obf1", "ai_obf2", "ai_obf3", "ai_obf4", "human"]
    prompt_parts = [
        f"{p}: acc={metrics[f'{p}_acc']:.3f} prob={metrics[f'{p}_mean_prob']:.3f}"
        for p in prompt_order if f"{p}_acc" in metrics
    ]
    if prompt_parts:
        print(f"  per-prompt:     {('  |  ').join(prompt_parts)}")

# ── 11. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    label = f"gw{GENRE_WEIGHT}"

    # Load tokenizer from first seed's saved copy (all seeds use the same tokenizer)
    tokenizer_path = f"{SAVE_PATH}_{label}_seed{SEEDS[0]}_tokenizer"
    if os.path.isdir(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"Saved tokenizer not found at {tokenizer_path}, loading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenise test set (or load cache)
    if os.path.exists(TEST_CACHE):
        print(f"Loading tokenised test set from cache: {TEST_CACHE}")
        test_dataset = AIDetectionDataset.from_cache(TEST_CACHE)
    else:
        print(f"Loading test data from {TEST_PATH}...")
        test_records = load_jsonl(TEST_PATH)
        print(f"  {len(test_records)} records")
        test_dataset = AIDetectionDataset(test_records, tokenizer)
        test_dataset.save(TEST_CACHE)
        print(f"  Cached to {TEST_CACHE}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Evaluate each seed's best checkpoint
    all_metrics: list = []

    for seed in SEEDS:
        ckpt_path = f"{SAVE_PATH}_{label}_seed{seed}.pt"
        print(f"\n{'=' * 60}")
        print(f"[{label}]  seed={seed}  |  checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found, skipping seed {seed}")
            continue

        model = MultiTaskModernBERT().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        metrics = evaluate(model, test_loader)
        all_metrics.append(metrics)

        print(f"\nTest results (seed={seed}):")
        print_metrics(metrics, prefix="  ")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        del model
        torch.cuda.empty_cache()

    # Aggregate across seeds
    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}")
        print(f"Aggregate test results [{label}]  |  seeds={SEEDS}")
        print(f"{'=' * 60}")
        all_keys = all_metrics[0].keys()
        agg = {}
        for k in all_keys:
            vals = [m[k] for m in all_metrics if k in m]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
            print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    else:
        agg = {k: v for k, v in all_metrics[0].items()} if all_metrics else {}

    # Save results
    results_path = f"{SAVE_PATH}_{label}_val_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "genre_weight": GENRE_WEIGHT,
            "seeds":        SEEDS,
            "per_seed":     {str(SEEDS[i]): m for i, m in enumerate(all_metrics)},
            "aggregate":    agg,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
