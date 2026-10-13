"""Shared data loading and splitting, so every model is evaluated on the same split.

The balancing and the 70/15/15 split reproduce train_model.py exactly
(same sampling, same random_state), so the TF-IDF + Random Forest baseline
and the transformer models are compared on identical test sets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "your_new_dataset.csv"
SEED = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["status"])
    X = df["statement"].fillna("").astype(str)
    y = df["status"].astype(str)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def balance(X, y, seed=SEED):
    """Undersample every class to the size of the smallest one (as in train_model.py)."""
    df = pd.DataFrame({"X": X, "y": y})
    min_count = df["y"].value_counts().min()
    df_balanced = pd.concat([
        df[df.y == label].sample(n=min_count, random_state=seed)
        for label in df["y"].unique()
    ])
    df_balanced = df_balanced.sample(frac=1, random_state=seed)
    return df_balanced.X, df_balanced.y


def split(X, y, seed=SEED):
    """Stratified 70 / 15 / 15 train / validation / test split (as in train_model.py)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_splits(setting):
    """setting = 'balanced' (undersampled, as in the original project) or 'full' (all data)."""
    X, y = load_data()
    if setting == "balanced":
        X, y = balance(X, y)
    elif setting != "full":
        raise ValueError("setting must be 'balanced' or 'full'")
    return split(X, y)
