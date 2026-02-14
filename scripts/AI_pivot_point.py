#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, classification_report, precision_recall_curve
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, Sequential

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pivot-bottom binary classification with CNN/LSTM/Transformer.")
    p.add_argument("--data-path", default="data/kospi200_20260212.csv")
    p.add_argument("--label-path", default="outputs/pivotal point/kospi200_labels_expanded.csv")
    p.add_argument("--seq-length", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-precision", type=float, default=0.30)
    return p.parse_args()


def load_and_preprocess_data(data_path: str, label_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(data_path, low_memory=False)
    df_label = pd.read_csv(label_path, low_memory=False)

    date_col = "날짜" if "날짜" in df_raw.columns else "date"
    if date_col not in df_raw.columns:
        raise ValueError("Date column not found in raw data.")

    label_col = "label_20d_expanded" if "label_20d_expanded" in df_label.columns else "label_20d"
    if label_col not in df_label.columns:
        raise ValueError("Label column (label_20d_expanded/label_20d) not found.")

    df = pd.merge(df_raw, df_label[[date_col, label_col]], on=date_col, how="inner")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Basic cleansing
    df = df[df["C"].notna() & (df["C"] != 0)].copy()
    df["O"] = df["O"].replace(0, np.nan).fillna(df["C"])
    df["H"] = df["H"].replace(0, np.nan).fillna(df["C"])
    df["L"] = df["L"].replace(0, np.nan).fillna(df["C"])
    valid = (
        (df["H"] >= df["L"])
        & (df["H"] >= df[["C", "O"]].max(axis=1))
        & (df["L"] <= df[["C", "O"]].min(axis=1))
    )
    df = df[valid].reset_index(drop=True)

    # Features
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

    # Binary target: bottom(-1)=1 else 0
    df["Target_Bottom"] = np.where(df[label_col] == -1, 1, 0)
    return df.dropna().reset_index(drop=True)


def build_datasets(df: pd.DataFrame, seq_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-8
    X_seq, y_lab, w_time = [], [], []
    time_weights = np.linspace(0.7, 1.0, len(df))

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
        X_seq.append(seq_features)
        y_lab.append(int(df.iloc[i + seq_length - 1]["Target_Bottom"]))
        w_time.append(float(time_weights[i + seq_length - 1]))

    X = np.array(X_seq)
    y = np.array(y_lab)
    w = np.array(w_time)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    w_train = w[:split_idx]

    # Merge time-weight and class-weight into single sample_weight.
    total = len(y_train)
    count_1 = max(1, int(np.sum(y_train)))
    count_0 = max(1, int(total - count_1))
    weight_0 = total / (2 * count_0)
    weight_1 = (total / (2 * count_1)) ** 0.5

    combined_w_train = np.zeros(len(y_train), dtype=float)
    for i in range(len(y_train)):
        class_w = weight_1 if y_train[i] == 1 else weight_0
        combined_w_train[i] = float(w_train[i]) * float(class_w)

    return X_train, X_test, y_train, y_test, combined_w_train


def build_cnn_1d(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation="relu"),
            Flatten(),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ],
        name="1D_CNN",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ],
        name="LSTM",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_transformer(input_shape: tuple[int, int]) -> Model:
    inputs = Input(shape=input_shape)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=32, num_heads=4, dropout=0.2)(x, x)
    x = Dropout(0.2)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(input_shape[-1])(x)
    x = x + res

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="Transformer")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    args = parse_args()
    df = load_and_preprocess_data(args.data_path, args.label_path)
    X_train, X_test, y_train, y_test, w_train = build_datasets(df, seq_length=int(args.seq_length))

    print(f"Train={X_train.shape}, Test={X_test.shape}, Bottom in test={int(np.sum(y_test))}/{len(y_test)}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    models = {
        "1D-CNN": build_cnn_1d(input_shape),
        "LSTM": build_lstm(input_shape),
        "Transformer": build_transformer(input_shape),
    }

    target_precision = float(args.target_precision)
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

        y_pred_prob = model.predict(X_test, verbose=0).flatten()

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recalls, precisions)
        print(f"[{name}] PR-AUC: {pr_auc:.4f}")

        optimal_threshold = 0.50
        for p, t in zip(precisions[:-1], thresholds):
            if p >= target_precision:
                optimal_threshold = float(t)
                break
        print(f"[{name}] target precision={target_precision:.2f} -> threshold={optimal_threshold:.4f}")

        y_pred_binary = (y_pred_prob >= optimal_threshold).astype(int)
        print(classification_report(y_test, y_pred_binary, target_names=["Not Bottom (0)", "Bottom (1)"], zero_division=0))


if __name__ == "__main__":
    main()
