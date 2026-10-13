"""Fine-tune a transformer (DistilBERT by default) on the shared splits.

Usage:  python train_transformer.py --setting balanced
        python train_transformer.py --setting full --model distilbert-base-uncased
"""
import argparse, json, os, time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from data_utils import get_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["balanced", "full"], default="balanced")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()
    set_seed(42)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(args.setting)
    labels = sorted(y_train.unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(X, y):
        ds = Dataset.from_dict({"text": list(X), "label": [label2id[v] for v in y]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(X_train, y_train), to_ds(X_val, y_val), to_ds(X_test, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "macro_f1": f1_score(p.label_ids, preds, average="macro")}

    out_dir = f"models/{args.model.split('/')[-1]}-{args.setting}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.06 * args.epochs * -(-len(train_ds) // args.batch)),  # 6% warmup
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    results = {"model": args.model, "setting": args.setting, "epochs": args.epochs,
               "lr": args.lr, "max_len": args.max_len, "n_train": len(X_train),
               "n_val": len(X_val), "n_test": len(X_test),
               "train_seconds": round(train_seconds, 1),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for name, ds, y in [("val", val_ds, y_val), ("test", test_ds, y_test)]:
        preds = np.argmax(trainer.predict(ds).predictions, axis=-1)
        pred_labels = [id2label[i] for i in preds]
        results[name] = {
            "accuracy": accuracy_score(list(y), pred_labels),
            "macro_f1": f1_score(list(y), pred_labels, average="macro"),
            "weighted_f1": f1_score(list(y), pred_labels, average="weighted"),
            "report": classification_report(list(y), pred_labels, output_dict=True),
        }
        if name == "test":
            results["labels"] = labels
            results["confusion_matrix"] = confusion_matrix(list(y), pred_labels, labels=labels).tolist()
        print(f"[{args.setting}] {name}: accuracy={results[name]['accuracy']:.4f} "
              f"macro-F1={results[name]['macro_f1']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/transformer_{args.model.split('/')[-1]}_{args.setting}.json", "w") as f:
        json.dump(results, f, indent=2)
    trainer.save_model(f"{out_dir}/best")
    tok.save_pretrained(f"{out_dir}/best")


if __name__ == "__main__":
    main()
