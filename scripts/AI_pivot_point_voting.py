#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.metrics import auc, classification_report, precision_recall_curve

from AI_pivot_point import build_datasets, build_lstm, build_transformer, load_and_preprocess_data

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM+Transformer and evaluate voting ensemble for pivot bottoms.")
    p.add_argument("--data-path", default="data/kospi200_20260212.csv")
    p.add_argument("--label-path", default="outputs/pivotal point/kospi200_labels_expanded.csv")
    p.add_argument("--seq-length", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-precision", type=float, default=0.30)
    p.add_argument("--voting", choices=["and", "or"], default="and")
    return p.parse_args()


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    threshold = 0.50
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            threshold = float(t)
            break
    return threshold, float(pr_auc)


def main() -> None:
    args = parse_args()

    df = load_and_preprocess_data(args.data_path, args.label_path)
    X_train, X_test, y_train, y_test, w_train = build_datasets(df, seq_length=int(args.seq_length))
    input_shape = (X_train.shape[1], X_train.shape[2])

    print(f"Train={X_train.shape}, Test={X_test.shape}, Bottom in test={int(np.sum(y_test))}/{len(y_test)}")

    print("\n[LSTM] training...")
    lstm_model = build_lstm(input_shape)
    lstm_model.fit(
        X_train,
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_split=0.2,
        sample_weight=w_train,
        verbose=0,
    )

    print("\n[Transformer] training...")
    trans_model = build_transformer(input_shape)
    trans_model.fit(
        X_train,
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_split=0.2,
        sample_weight=w_train,
        verbose=0,
    )

    prob_lstm = lstm_model.predict(X_test, verbose=0).flatten()
    prob_trans = trans_model.predict(X_test, verbose=0).flatten()

    th_lstm, pr_auc_lstm = _optimal_threshold(y_test, prob_lstm, float(args.target_precision))
    th_trans, pr_auc_trans = _optimal_threshold(y_test, prob_trans, float(args.target_precision))

    pred_lstm = (prob_lstm >= th_lstm).astype(int)
    pred_trans = (prob_trans >= th_trans).astype(int)

    if args.voting == "and":
        y_pred_ens = pred_lstm & pred_trans
    else:
        y_pred_ens = pred_lstm | pred_trans

    print(f"\n[LSTM] PR-AUC={pr_auc_lstm:.4f}, threshold={th_lstm:.4f}")
    print(f"[Transformer] PR-AUC={pr_auc_trans:.4f}, threshold={th_trans:.4f}")
    print(f"[Ensemble] voting={args.voting.upper()}, target_precision={float(args.target_precision):.2f}")
    print(classification_report(y_test, y_pred_ens, target_names=["Not Bottom (0)", "Bottom (1)"], zero_division=0))


if __name__ == "__main__":
    main()
