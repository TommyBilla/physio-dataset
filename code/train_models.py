#!/usr/bin/env python3
"""
train_models.py  —  runs locally or pasted into Colab.

One model per exercise, multi-label (errors co-occur, so each label gets its
own binary head).

Evaluation is GroupKFold on source_recording_id. This is not optional: five
clips cut from the same video share the actor's exact posture, framing and
lighting for that take. Split them across train and test and the score measures
memorisation, not generalisation.

Reports per-label precision / recall / F1 / support, plus macro-F1. Accuracy is
deliberately absent — with 5-20 positives out of 45 clips, predicting all-zero
scores 80% accuracy and is worthless.

Usage:
    python train_models.py train_data/
    python train_models.py train_data/ --model xgb
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

META = {"source_recording_id", "subject_id"}
SEED = 42

# below this many groups, cross-validation cannot produce a meaningful estimate
MIN_GROUPS = 4


def make_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,          # sized for ~40 rows, not 24k
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            eval_metric="logloss",
        )
    raise ValueError(kind)


def split_frame(df: pd.DataFrame):
    """Separate the frame into features, labels and grouping key."""
    labels = [c for c in df.columns
              if df[c].dropna().isin([0, 1]).all() and c not in META]
    feats = [c for c in df.columns if c not in labels and c not in META]
    return df[feats], df[labels], df["source_recording_id"]


def evaluate(X, y, groups, kind):
    """Out-of-fold predictions for every label, via GroupKFold."""
    n_groups = groups.nunique()
    k = min(5, n_groups)
    gkf = GroupKFold(n_splits=k)

    oof = pd.DataFrame(0, index=y.index, columns=y.columns)

    for tr, te in gkf.split(X, groups=groups):
        for lab in y.columns:
            ytr = y[lab].iloc[tr]
            if ytr.nunique() < 2:
                continue  # this fold saw only one class; leave prediction at 0
            m = make_model(kind)
            m.fit(X.iloc[tr], ytr)
            oof.loc[y.index[te], lab] = m.predict(X.iloc[te])

    rows = []
    for lab in y.columns:
        p, r, f, _ = precision_recall_fscore_support(
            y[lab], oof[lab], average="binary", zero_division=0
        )
        rows.append({
            "label": lab,
            "support": int(y[lab].sum()),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
        })
    return pd.DataFrame(rows), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir", type=Path)
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = ap.parse_args()

    files = sorted(args.datadir.glob("train_*.parquet"))
    if not files:
        raise SystemExit(f"no train_*.parquet in {args.datadir}")

    macro = []
    for f in files:
        ex = f.stem.replace("train_", "")
        df = pd.read_parquet(f)
        X, y, groups = split_frame(df)
        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        n_groups = groups.nunique()
        header = f"=== {ex}: {len(df)} clips, {n_groups} groups, {X.shape[1]} features"

        if n_groups < MIN_GROUPS:
            print(header)
            print(f"  SKIPPED — only {n_groups} recording groups. Cross-validation "
                  f"here would report a number, not an estimate. Collect more "
                  f"recordings before training this exercise.\n")
            continue

        res, k = evaluate(X, y, groups, args.model)
        print(header + f", GroupKFold k={k}")
        print(res.to_string(index=False))
        mf1 = res["f1"].mean()
        print(f"  macro-F1: {mf1:.3f}")
        if res["support"].min() < 8:
            print(f"  note: smallest label has {res['support'].min()} positives — "
                  f"its F1 swings on one or two clips")
        print()
        macro.append({"exercise": ex, "macro_f1": round(mf1, 3), "k": k})

    if macro:
        print("== summary ==")
        print(pd.DataFrame(macro).to_string(index=False))


if __name__ == "__main__":
    main()
