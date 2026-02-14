#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM model for pivot-point classification.")
    p.add_argument("--data-path", default="data/kospi200_20260212.csv")
    p.add_argument("--label-path", default="outputs/pivotal point/kospi200_labels_expanded.csv")
    p.add_argument("--label-col", default="label_20d_expanded")
    p.add_argument("--seq-length", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold-bottom", type=float, default=0.80)
    p.add_argument("--threshold-top", type=float, default=0.80)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_raw = pd.read_csv(args.data_path, low_memory=False)
    df_label = pd.read_csv(args.label_path, low_memory=False)

    label_col = args.label_col
    if label_col not in df_label.columns:
        fallback = "label_20d"
        if fallback in df_label.columns:
            label_col = fallback
        else:
            raise ValueError(f"Label column not found: {args.label_col}")

    df = pd.merge(df_raw, df_label[["날짜", label_col]], on="날짜", how="inner")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜").reset_index(drop=True)

    eps = 1e-8
    df["Candle_Len"] = df["H"] - df["L"] + eps
    df["Body_Len"] = (df["C"] - df["O"]).abs()
    df["Upper_Tail"] = df["H"] - df[["C", "O"]].max(axis=1)
    df["Lower_Tail"] = df[["C", "O"]].min(axis=1) - df["L"]
    df["Body_Ratio"] = df["Body_Len"] / df["Candle_Len"]
    df["Upper_Tail_Ratio"] = df["Upper_Tail"] / df["Candle_Len"]
    df["Lower_Tail_Ratio"] = df["Lower_Tail"] / df["Candle_Len"]

    df["Prev_Close"] = df["C"].shift(1)
    df["TR1"] = df["H"] - df["L"]
    df["TR2"] = (df["H"] - df["Prev_Close"]).abs()
    df["TR3"] = (df["L"] - df["Prev_Close"]).abs()
    df["TR"] = df[["TR1", "TR2", "TR3"]].max(axis=1)
    df["ATR_10"] = df["TR"].rolling(window=10).mean()
    df = df.dropna().reset_index(drop=True)

    seq_length = int(args.seq_length)
    X_sequences: list[np.ndarray] = []
    y_labels: list[list[int]] = []
    sample_weights: list[float] = []
    weights_array = np.linspace(0.7, 1.0, len(df))

    for i in range(len(df) - seq_length):
        window = df.iloc[i : i + seq_length]
        local_max = window["H"].max()
        local_min = window["L"].min()
        price_range = local_max - local_min + eps

        scaled_ohlc = (window[["O", "H", "L", "C"]] - local_min) / price_range
        scaled_v = (window["V"] - window["V"].min()) / (window["V"].max() - window["V"].min() + eps)
        scaled_atr = (window["ATR_10"] - window["ATR_10"].min()) / (window["ATR_10"].max() - window["ATR_10"].min() + eps)

        seq_features = np.column_stack(
            (
                scaled_ohlc.values,
                scaled_v.values,
                window["Body_Ratio"].values,
                window["Upper_Tail_Ratio"].values,
                window["Lower_Tail_Ratio"].values,
                scaled_atr.values,
            )
        )
        X_sequences.append(seq_features)

        yv = int(window.iloc[-1][label_col])
        if yv == -1:
            y_labels.append([1, 0, 0])
        elif yv == 0:
            y_labels.append([0, 1, 0])
        else:
            y_labels.append([0, 0, 1])
        sample_weights.append(float(weights_array[i + seq_length - 1]))

    X = np.array(X_sequences)
    y = np.array(y_labels)
    w = np.array(sample_weights)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    w_train = w[:split_idx]

    total = len(y_train)
    class_weights = {
        0: total / (3 * max(1, np.sum(y_train[:, 0]))),
        1: total / (3 * max(1, np.sum(y_train[:, 1]))),
        2: total / (3 * max(1, np.sum(y_train[:, 2]))),
    }

    # Keras does not support simultaneous use of sample_weight and class_weight.
    # Merge both effects into one per-sample weight vector.
    combined_w_train = np.zeros(len(y_train), dtype=float)
    for i in range(len(y_train)):
        true_class = int(np.argmax(y_train[i]))  # 0=Bottom, 1=Normal, 2=Top
        combined_w_train[i] = float(w_train[i]) * float(class_weights[true_class])

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        validation_split=0.2,
        sample_weight=combined_w_train,
        verbose=1,
    )

    y_pred_prob = model.predict(X_test)
    threshold_bottom = float(args.threshold_bottom)
    threshold_top = float(args.threshold_top)

    y_pred_classes: list[int] = []
    for prob in y_pred_prob:
        # prob[0]=Bottom, prob[1]=Normal, prob[2]=Top
        if prob[0] >= threshold_bottom:
            y_pred_classes.append(0)
        elif prob[2] >= threshold_top:
            y_pred_classes.append(2)
        else:
            y_pred_classes.append(1)
    y_pred_classes = np.array(y_pred_classes)
    y_true_classes = np.argmax(y_test, axis=1)
    print(f"\n[ Threshold evaluation ] bottom={threshold_bottom:.2f}, top={threshold_top:.2f}")
    print(classification_report(y_true_classes, y_pred_classes, target_names=["Bottom (-1)", "Normal (0)", "Top (1)"]))


if __name__ == "__main__":
    main()
