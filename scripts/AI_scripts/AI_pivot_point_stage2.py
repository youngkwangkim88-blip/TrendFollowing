#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, precision_recall_curve
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential

from AI_pivot_point import build_datasets, load_and_preprocess_data

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 pivot-bottom models (CNN-LSTM, Encoder-Classifier).")
    p.add_argument("--data-path", default="data/kospi200_20260212.csv")
    p.add_argument("--label-path", default="outputs/pivotal point/kospi200_labels_expanded.csv")
    p.add_argument("--seq-length", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-precision", type=float, default=0.30)
    p.add_argument("--out-csv", default="outputs/pivotal point/ai_pivot_stage2_results.csv")
    return p.parse_args()


def build_cnn_lstm(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ],
        name="CNN_LSTM_Hybrid",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_encoder_classifier(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(8, activation="relu", name="bottleneck_latent_space"),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ],
        name="Encoder_Bottleneck_Classifier",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _evaluate_dynamic_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float
) -> tuple[float, float, np.ndarray]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    opt_th = 0.50
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            opt_th = float(t)
            break
    y_pred = (y_prob >= opt_th).astype(int)
    return pr_auc, opt_th, y_pred


def main() -> None:
    args = parse_args()
    df = load_and_preprocess_data(args.data_path, args.label_path)
    X_train, X_test, y_train, y_test, w_train = build_datasets(df, seq_length=int(args.seq_length))
    input_shape = (X_train.shape[1], X_train.shape[2])

    print(f"Train={X_train.shape}, Test={X_test.shape}, Bottom in test={int(np.sum(y_test))}/{len(y_test)}")

    models = {
        "CNN-LSTM": build_cnn_lstm(input_shape),
        "Encoder-Classifier": build_encoder_classifier(input_shape),
    }
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        print(f"\n{'=' * 60}\n[{name}] training...")
        model.fit(
            X_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            validation_split=0.2,
            sample_weight=w_train,
            verbose=0,
        )

        y_prob = model.predict(X_test, verbose=0).flatten()
        pr_auc, opt_th, y_pred = _evaluate_dynamic_threshold(
            y_true=y_test,
            y_prob=y_prob,
            target_precision=float(args.target_precision),
        )
        acc = float(np.mean(y_pred == y_test))
        pred_bottom = int(np.sum(y_pred))
        true_bottom = int(np.sum(y_test))
        rows.append(
            {
                "model": name,
                "pr_auc": float(pr_auc),
                "optimal_threshold": float(opt_th),
                "accuracy": acc,
                "pred_bottom_count": pred_bottom,
                "true_bottom_count": true_bottom,
            }
        )
        print(f"[{name}] PR-AUC: {pr_auc:.4f}")
        print(f"[{name}] Optimal threshold (target precision={float(args.target_precision):.2f}): {opt_th:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Bottom (0)", "Bottom (1)"], zero_division=0))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved stage2 summary: {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
