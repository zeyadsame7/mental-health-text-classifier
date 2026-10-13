# Results

All models are evaluated on the **same held-out test set** for each setting (stratified 70/15/15 split, seed 42).

| Setting | Model | Train size | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|
| balanced | TF-IDF + Random Forest | 5,884 | 67.3% | 0.670 | 0.670 |
| balanced | DistilBERT (fine-tuned) | 5,884 | 76.0% | 0.759 | 0.759 |
| full | TF-IDF + Random Forest | 37,130 | 70.9% | 0.621 | 0.694 |
| full | DistilBERT (fine-tuned) | 37,130 | 82.7% | 0.793 | 0.828 |
| full | RoBERTa-base (fine-tuned) | 37,130 | 85.1% | 0.822 | 0.851 |

## Per-class results: RoBERTa-base (fine-tuned) (full dataset)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anxiety | 0.89 | 0.87 | 0.88 | 584 |
| Bipolar | 0.79 | 0.86 | 0.82 | 431 |
| Depression | 0.82 | 0.79 | 0.80 | 2311 |
| Normal | 0.97 | 0.97 | 0.97 | 2453 |
| Personality disorder | 0.79 | 0.66 | 0.72 | 180 |
| Stress | 0.79 | 0.80 | 0.80 | 400 |
| Suicidal | 0.75 | 0.78 | 0.76 | 1598 |
