"""Collect results/*.json into results/RESULTS.md and comparison / confusion-matrix charts."""
import glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"TF-IDF + Random Forest": "TF-IDF + Random Forest",
         "distilbert-base-uncased": "DistilBERT (fine-tuned)",
         "roberta-base": "RoBERTa-base (fine-tuned)"}

runs = [json.load(open(p)) for p in sorted(glob.glob("results/*.json"))]
runs.sort(key=lambda r: (r["setting"] != "balanced", list(NAMES).index(r["model"])))

lines = ["# Results", "",
         "All models are evaluated on the **same held-out test set** for each setting "
         "(stratified 70/15/15 split, seed 42).", "",
         "| Setting | Model | Train size | Test accuracy | Test macro-F1 | Test weighted-F1 |",
         "|---|---|---|---|---|---|"]
for r in runs:
    t = r["test"]
    lines.append(f"| {r['setting']} | {NAMES[r['model']]} | {r['n_train']:,} | "
                 f"{t['accuracy']:.1%} | {t['macro_f1']:.3f} | {t['weighted_f1']:.3f} |")

best = max((r for r in runs if r["setting"] == "full"), key=lambda r: r["test"]["macro_f1"])
lines += ["", f"## Per-class results: {NAMES[best['model']]} (full dataset)", "",
          "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
for c in best["labels"]:
    m = best["test"]["report"][c]
    lines.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1-score']:.2f} | {int(m['support'])} |")
open("results/RESULTS.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# Comparison chart
fig, ax = plt.subplots(figsize=(8, 4.5))
settings = ["balanced", "full"]
models = [m for m in NAMES if any(r["model"] == m for r in runs)]
w = 0.8 / len(models)
for i, m in enumerate(models):
    xs, vals = [], []
    for j, s in enumerate(settings):
        v = next((r["test"]["macro_f1"] for r in runs if r["model"] == m and r["setting"] == s), None)
        if v is not None:  # only draw runs that exist
            xs.append(j + i * w); vals.append(v)
    bars = ax.bar(xs, vals, w, label=NAMES[m])
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_xticks(np.arange(len(settings)) + w * (len(models) - 1) / 2)
ax.set_xticklabels(["Balanced (8.4K samples)", "Full dataset (53K samples)"])
ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1.12)
ax.legend(loc="upper center", ncol=len(models), fontsize=9, frameon=False)
ax.set_title("Mental-health text classification: baseline vs. fine-tuned transformers")
fig.tight_layout(); fig.savefig("results/comparison.png", dpi=150)

# Confusion matrix of the best model
cm = np.array(best["confusion_matrix"]); cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(best["labels"]))); ax.set_xticklabels(best["labels"], rotation=40, ha="right")
ax.set_yticks(range(len(best["labels"]))); ax.set_yticklabels(best["labels"])
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{NAMES[best['model']]}: normalized confusion matrix (test)")
fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=150)
print("\n".join(lines))
