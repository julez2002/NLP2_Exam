import subprocess
subprocess.run(
    [
        "pip", "install", "-q",
        "numpy",
        "transformers",
        "accelerate",
        "scikit-learn",
    ],
    check=True,
)
# Imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, silhouette_score
from tqdm.auto import tqdm

# Config
MODEL_NAME   = "answerdotai/ModernBERT-base"
TRAIN_PATH   = 'merged_train.jsonl'
VAL_PATH     = 'val.jsonl'
SAVE_PATH    = 'baseline'

MAX_LEN      = 4096  
LR           = 2e-5
EPOCHS       = 15     
BATCH_SIZE   = 16     
ACCUM_STEPS  = 4      
PATIENCE     = 4      
MIN_DELTA    = 5e-3   
SEEDS        = [42, 123, 7]   

_cache_tag   = f"{MODEL_NAME.replace('/', '_')}_{MAX_LEN}"
TRAIN_CACHE  = f'train_tokenized_{_cache_tag}.pt'   
VAL_CACHE    = f'val_tokenized_{_cache_tag}.pt'

GENRE_MAP = {"fiction": 0, "news": 1, "essays": 2}

# Device
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data Utilities
def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_binary_weights(records: list) -> torch.Tensor:
    n       = len(records)
    n_human = sum(1 for r in records if r["label"] == 0)
    n_ai    = n - n_human
    return torch.tensor([n / (2 * n_human), n / (2 * n_ai)], dtype=torch.float)



# Dataset
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
            })

    def save(self, path: str) -> None:
        torch.save(self.data, path)

    @classmethod
    def from_cache(cls, path: str) -> "AIDetectionDataset":
        obj      = cls.__new__(cls)
        obj.data = torch.load(path, weights_only=False)
        return obj

    def sort_by_length(self, noise: int = 8) -> None:
        self.data.sort(
            key=lambda x: len(x["input_ids"]) + random.randint(0, noise)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

# Collate Function
def make_collate_fn(pad_token_id: int):
    
    def collate_fn(batch: list) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, attention_mask = [], []
        source_labels, genre_labels, attacks = [], [], []

        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"]      + [pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            source_labels.append(x["source_label"])
            genre_labels.append(x["genre_label"])
            attacks.append(x["attack"])

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "source_label":   torch.tensor(source_labels,  dtype=torch.long),
            "genre_label":    torch.tensor(genre_labels,   dtype=torch.long),
            "attack":         attacks,   # list[str | None]
        }
    return collate_fn

# Model
class ModernBERTClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size  
        self.dropout     = nn.Dropout(0.1)
        self.source_head = nn.Linear(hidden, 2)

    @staticmethod
    def mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.mean_pool(out.last_hidden_state, attention_mask))
        if return_embeddings:
            return self.source_head(pooled), pooled
        return self.source_head(pooled)

# Evaluation 
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    all_genre_labels: list = []
    all_attacks: list = []
    all_embeddings:    list = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            source_logits, pooled = model(ids, mask, return_embeddings=True)

        all_probs.extend(F.softmax(source_logits, dim=-1)[:, 1].cpu().tolist())
        all_preds.extend(source_logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["source_label"].tolist())
        all_genre_labels.extend(batch["genre_label"].tolist())
        all_attacks.extend(batch["attack"])
        all_embeddings.append(pooled.float().cpu())

    _fprs, _tprs, _threshs = roc_curve(all_labels, all_probs)
    _valid = np.where(_fprs <= 0.05)[0]
    _t5_thresh = float(_threshs[_valid[-1]]) if len(_valid) else 1.0

    metrics = {
        "binary_f1":      f1_score(all_labels, all_preds, average="binary", pos_label=1),
        "macro_f1":       f1_score(all_labels, all_preds, average="macro"),
        "auc_roc":        roc_auc_score(all_labels, all_probs),
        "accuracy":       accuracy_score(all_labels, all_preds),
        "tpr_at_5fpr":    float(_tprs[_valid[-1]]) if len(_valid) else 0.0,
    }

    _emb = torch.cat(all_embeddings, 0).numpy()
    _sil_n = min(len(all_labels), 2000)
    metrics["source_silhouette"] = float(silhouette_score(
        _emb, all_labels, metric="cosine", sample_size=_sil_n, random_state=0,
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

    return metrics

# Training
def train(
    model: nn.Module,
    train_dataset: AIDetectionDataset,
    val_loader: DataLoader,
    binary_weights: torch.Tensor,
    tokenizer,
    save_path: str = SAVE_PATH,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    batches_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    updates_per_epoch = batches_per_epoch // ACCUM_STEPS
    total_steps       = updates_per_epoch * EPOCHS
    warmup_steps      = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    source_ce = nn.CrossEntropyLoss(weight=binary_weights.to(device))
    best_val_macro_f1 = 0.0
    patience_counter  = 0
    history: list     = []

    for epoch in range(EPOCHS):
        train_dataset.sort_by_length(noise=8)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer.pad_token_id),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            s_labs = batch["source_label"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                source_logits = model(ids, mask)
                loss          = source_ce(source_logits, s_labs) / ACCUM_STEPS

            loss.backward()
            running_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        if (step + 1) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        n_steps  = len(train_loader)
        avg_loss = running_loss / n_steps
        metrics  = evaluate(model, val_loader)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}  |  loss={avg_loss:.4f}"
            f"  |  bin_f1={metrics['binary_f1']:.4f}"
            f"  |  macro_f1={metrics['macro_f1']:.4f}"
            f"  |  auc_roc={metrics['auc_roc']:.4f}"
            f"  |  acc={metrics['accuracy']:.4f}"
            f"  |  src_sil={metrics['source_silhouette']:.4f}"
        )
        if "clean_binary_f1" in metrics:
            clean_line = f"           clean_f1={metrics['clean_binary_f1']:.4f}"
            if "clean_auc_roc"     in metrics: clean_line += f"  |  clean_auc={metrics['clean_auc_roc']:.4f}"
            if "clean_tpr_at_5fpr" in metrics: clean_line += f"  |  clean_tpr@5fpr={metrics['clean_tpr_at_5fpr']:.4f}"
            print(clean_line)
        if "attacked_binary_f1" in metrics:
            atk_line = f"           attacked_f1={metrics['attacked_binary_f1']:.4f}"
            if "attacked_auc_roc"     in metrics: atk_line += f"  |  attacked_auc={metrics['attacked_auc_roc']:.4f}"
            if "attacked_tpr_at_5fpr" in metrics: atk_line += f"  |  attacked_tpr@5fpr={metrics['attacked_tpr_at_5fpr']:.4f}"
            print(atk_line)
        genre_f1_parts = "  |  ".join(
            f"{g}_f1={metrics[f'{g}_binary_f1']:.4f}"
            for g in ("fiction", "news", "essays")
            if f"{g}_binary_f1" in metrics
        )
        if genre_f1_parts:
            print(f"           per-genre binary: {genre_f1_parts}")

        val_macro_f1 = metrics["macro_f1"]
        prev_best    = best_val_macro_f1
        saved        = False

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            saved = True
            torch.save(model.state_dict(), save_path + ".pt")
            tokenizer.save_pretrained(save_path + "_tokenizer")
            with open(save_path + "_config.json", "w") as _f:
                json.dump({
                    "encoder":            MODEL_NAME,
                    "hidden_size":        model.encoder.config.hidden_size,
                    "num_source_classes": model.source_head.out_features,
                    "pooling":            "masked_mean",
                    "dropout":            model.dropout.p,
                    "max_len":            MAX_LEN,
                }, _f, indent=2)
            print(f"  Saved best model  (val macro_f1={best_val_macro_f1:.4f})")

        if val_macro_f1 > prev_best + MIN_DELTA:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No significant improvement ({patience_counter}/{PATIENCE})")

        history.append({
            "epoch":            epoch + 1,
            "train_loss":       avg_loss,
            "saved_checkpoint": saved,
            "patience_counter": patience_counter,
            **metrics,
        })

        if patience_counter >= PATIENCE:
            print("  Early stopping triggered (F1 plateaued or decreased).")
            break

    with open(save_path + "_history.json", "w") as _f:
        json.dump(history, _f, indent=2)

    return best_val_macro_f1

# Seed runner
def run_seeds(
    label: str,
    train_dataset: AIDetectionDataset,
    val_loader: DataLoader,
    binary_weights: torch.Tensor,
    tokenizer,
) -> dict:
    """Train across all SEEDS. Returns {seed: metrics}."""
    results: dict = {}
    all_run_metrics: list = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"[{label}]  seed={seed}")
        print(f"{'=' * 60}\n")
        sys.stdout.flush()

        set_seed(seed)
        save_path = f"{SAVE_PATH}_{label}_seed{seed}"

        model = ModernBERTClassifier().to(device)

        print("Warming up CUDA kernels...")
        sys.stdout.flush()
        with torch.no_grad():
            dummy_ids  = torch.zeros(1, 64, dtype=torch.long, device=device)
            dummy_mask = torch.ones(1, 64, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _ = model(dummy_ids, dummy_mask)
        print("Warmup done.\n")
        sys.stdout.flush()

        best_val_f1 = train(
            model, train_dataset, val_loader,
            binary_weights, tokenizer,
            save_path=save_path,
        )
        print(f"\nBest val macro F1 ({label}, seed={seed}): {best_val_f1:.4f}")

        print("\nLoading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(save_path + ".pt", map_location=device, weights_only=True))
        final_metrics = evaluate(model, val_loader)
        results[seed]  = final_metrics
        all_run_metrics.append(final_metrics)

        print(f"\nFinal evaluation ({label}, seed={seed}):")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Summary [{label}]: seeds={SEEDS}")
    print(f"{'=' * 60}")
    for k in all_run_metrics[0]:
        vals = [m[k] for m in all_run_metrics if k in m]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}" if len(vals) > 1 else f"  {k}: {vals[0]:.4f}")

    return results


# Main
def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading data...")
    train_records = load_jsonl(TRAIN_PATH)
    val_records   = load_jsonl(VAL_PATH)
    print(f"  train={len(train_records)} | val={len(val_records)}")

    binary_weights = compute_binary_weights(train_records)
    print(f"  Binary weights — human: {binary_weights[0]:.3f} | AI: {binary_weights[1]:.3f}")

    if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE):
        print("\nLoading tokenised datasets from cache...")
        train_dataset = AIDetectionDataset.from_cache(TRAIN_CACHE)
        val_dataset   = AIDetectionDataset.from_cache(VAL_CACHE)
        print(f"  train={len(train_dataset)} | val={len(val_dataset)}")
    else:
        print("\nTokenising (done once — will be cached to disk)...")
        train_dataset = AIDetectionDataset(train_records, tokenizer)
        val_dataset   = AIDetectionDataset(val_records,   tokenizer)
        train_dataset.save(TRAIN_CACHE)
        val_dataset.save(VAL_CACHE)
        print(f"  Cached to {TRAIN_CACHE} and {VAL_CACHE}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    kwargs = dict(
        train_dataset=train_dataset, val_loader=val_loader,
        binary_weights=binary_weights,
        tokenizer=tokenizer,
    )

    target_results = run_seeds("baseline", **kwargs)

    results_path = f"{SAVE_PATH}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "seeds":   SEEDS,
            "results": {str(s): v for s, v in target_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")



if __name__ == "__main__":
    main()
