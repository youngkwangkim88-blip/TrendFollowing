#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import argrelextrema
from sklearn.metrics import auc, classification_report, precision_recall_curve
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
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
    p = argparse.ArgumentParser(description="Stage4 multi-ticker pivot-bottom benchmark")
    p.add_argument("--panel-csv", default="data/krx100_adj_5000.csv")
    p.add_argument("--symbols", default="005930,000660", help="comma-separated ticker list")
    p.add_argument("--seq-length", type=int, default=60)
    p.add_argument("--label-order", type=int, default=10, help="argrelextrema order for local minima")
    p.add_argument("--label-expand", type=int, default=1, help="neighbor expansion around pivot")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--target-precision", type=float, default=0.30)
    p.add_argument("--out-csv", default="outputs/pivotal point/ai_pivot_stage4_results.csv")
    p.add_argument("--summary-csv", default="outputs/pivotal point/ai_pivot_stage4_summary.csv")
    return p.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. candidates={candidates}, columns={list(df.columns)}")
    return None


def _normalize_symbol(sym: object) -> str:
    s = str(sym).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s


def _parse_symbols(raw: str) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return sorted({_normalize_symbol(x) for x in vals})


def _prepare_symbol_frame(df: pd.DataFrame, seq_length: int, label_order: int, label_expand: int) -> pd.DataFrame:
    epsilon = 1e-8

    date_col = _pick_col(df, ["날짜", "date", "Date", "?좎쭨"])
    open_col = _pick_col(df, ["O", "open", "Open"])
    high_col = _pick_col(df, ["H", "high", "High"])
    low_col = _pick_col(df, ["L", "low", "Low"])
    close_col = _pick_col(df, ["C", "close", "Close"])
    volume_col = _pick_col(df, ["V", "volume", "Volume"], required=False)

    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in [open_col, high_col, low_col, close_col]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if volume_col is not None:
        x[volume_col] = pd.to_numeric(x[volume_col], errors="coerce").fillna(0.0)
    else:
        x["_volume"] = 0.0
        volume_col = "_volume"

    x = x[x[close_col].notna() & (x[close_col] > 0)].copy()
    for c in [open_col, high_col, low_col]:
        x.loc[x[c] <= 0, c] = np.nan
        x[c] = x[c].fillna(x[close_col])

    x[high_col] = x[[high_col, open_col, close_col]].max(axis=1)
    x[low_col] = x[[low_col, open_col, close_col]].min(axis=1)

    x["Candle_Len"] = x[high_col] - x[low_col] + epsilon
    x["Body_Len"] = (x[close_col] - x[open_col]).abs()
    x["Upper_Tail"] = x[high_col] - x[[close_col, open_col]].max(axis=1)
    x["Lower_Tail"] = x[[close_col, open_col]].min(axis=1) - x[low_col]
    x["Body_Ratio"] = x["Body_Len"] / x["Candle_Len"]
    x["Upper_Tail_Ratio"] = x["Upper_Tail"] / x["Candle_Len"]
    x["Lower_Tail_Ratio"] = x["Lower_Tail"] / x["Candle_Len"]

    x["Prev_Close"] = x[close_col].shift(1)
    x["TR1"] = x[high_col] - x[low_col]
    x["TR2"] = (x[high_col] - x["Prev_Close"]).abs()
    x["TR3"] = (x[low_col] - x["Prev_Close"]).abs()
    x["TR"] = x[["TR1", "TR2", "TR3"]].max(axis=1)
    x["ATR_10"] = x["TR"].rolling(window=10).mean()

    x["Target_Bottom"] = 0
    mins = argrelextrema(x[low_col].values, np.less_equal, order=int(label_order))[0]
    max_idx = len(x) - 1
    for idx in mins:
        lo = max(0, int(idx) - int(label_expand))
        hi = min(max_idx, int(idx) + int(label_expand))
        x.loc[lo:hi, "Target_Bottom"] = 1

    x = x.dropna().reset_index(drop=True)
    if len(x) <= seq_length + 10:
        raise ValueError("Not enough rows after preprocessing")

    x = x.rename(columns={open_col: "O", high_col: "H", low_col: "L", close_col: "C", volume_col: "V"})
    return x


def _build_sequences(df: pd.DataFrame, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1e-8
    features_to_scale = ["O", "H", "L", "C"]

    X_seq: list[np.ndarray] = []
    y_lab: list[float] = []

    for i in range(len(df) - seq_length):
        window = df.iloc[i : i + seq_length]
        local_min_val = float(window["L"].min())
        price_range = float(window["H"].max()) - local_min_val + epsilon

        scaled_ohlc = (window[features_to_scale] - local_min_val) / price_range
        scaled_v = (window["V"] - window["V"].min()) / (window["V"].max() - window["V"].min() + epsilon)
        scaled_atr = (window["ATR_10"] - window["ATR_10"].min()) / (window["ATR_10"].max() - window["ATR_10"].min() + epsilon)

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
        y_lab.append(float(df.iloc[i + seq_length - 1]["Target_Bottom"]))

    return np.array(X_seq, dtype=np.float32), np.array(y_lab, dtype=np.int32)


def _split_train_test(X: np.ndarray, y: np.ndarray, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(X) * ratio)
    return X[:split], X[split:], y[:split], y[split:]


def load_multi_ticker_dataset(
    panel_csv: str,
    symbols: list[str],
    seq_length: int,
    label_order: int,
    label_expand: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(panel_csv, dtype={"ticker": str}, low_memory=False)
    ticker_col = _pick_col(raw, ["ticker", "종목코드", "symbol"])
    raw[ticker_col] = raw[ticker_col].astype(str).map(_normalize_symbol)

    X_train_list: list[np.ndarray] = []
    X_test_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []

    for sym in symbols:
        g = raw[raw[ticker_col] == sym].copy()
        if g.empty:
            print(f"[SKIP] symbol={sym}: no rows in panel")
            continue
        try:
            prepared = _prepare_symbol_frame(g, seq_length=seq_length, label_order=label_order, label_expand=label_expand)
            X, y = _build_sequences(prepared, seq_length=seq_length)
            if len(X) < 100:
                print(f"[SKIP] symbol={sym}: too few sequences ({len(X)})")
                continue
            xtr, xte, ytr, yte = _split_train_test(X, y, ratio=0.8)
            X_train_list.append(xtr)
            X_test_list.append(xte)
            y_train_list.append(ytr)
            y_test_list.append(yte)
            print(f"[OK] {sym}: train={len(xtr)}, test={len(xte)}, bottom_train={int(np.sum(ytr))}")
        except Exception as e:
            print(f"[SKIP] symbol={sym}: {e}")

    if not X_train_list:
        raise SystemExit("No valid symbol dataset created. Check --symbols and panel columns.")

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    total = len(y_train)
    count_1 = int(np.sum(y_train))
    count_0 = int(total - count_1)
    if count_1 == 0 or count_0 == 0:
        w_train = np.ones_like(y_train, dtype=np.float32)
    else:
        weight_0 = total / (2.0 * count_0)
        weight_1 = (total / (2.0 * count_1)) ** 0.5
        w_train = np.where(y_train == 1, weight_1, weight_0).astype(np.float32)

    return X_train, X_test, y_train, y_test, w_train


def build_model(model_name: str, input_shape: tuple[int, int]) -> Model:
    if model_name == "CNN-LSTM":
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(64, 3, activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling1D(2),
                Conv1D(128, 3, activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling1D(2),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ],
            name="CNN_LSTM_60d",
        )
    elif model_name == "Encoder":
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(64, return_sequences=True),
                LSTM(32, return_sequences=False),
                Dense(16, activation="relu"),
                Dense(8, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ],
            name="Encoder_60d",
        )
    elif model_name == "Transformer":
        inputs = Input(shape=input_shape)
        x = Dense(32)(inputs)
        pos = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_emb = Embedding(input_dim=input_shape[0], output_dim=32)(pos)
        x = x + pos_emb
        res = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(key_dim=32, num_heads=4, dropout=0.3)(x, x)
        x = Dropout(0.3)(x)
        x = x + res
        res = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(32)(x)
        x = x + res
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs, name="Transformer_60d")
    else:
        raise ValueError(model_name)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> tuple[float, float, np.ndarray]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    opt_th = 0.50
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            opt_th = float(t)
            break

    y_pred = (y_prob >= opt_th).astype(int)
    return float(pr_auc), float(opt_th), y_pred


def main() -> None:
    args = parse_args()
    symbols = _parse_symbols(args.symbols)

    print("Loading multi-ticker dataset...")
    X_train, X_test, y_train, y_test, w_train = load_multi_ticker_dataset(
        panel_csv=args.panel_csv,
        symbols=symbols,
        seq_length=int(args.seq_length),
        label_order=int(args.label_order),
        label_expand=int(args.label_expand),
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Train={X_train.shape}, Test={X_test.shape}, Bottom in test={int(np.sum(y_test))}/{len(y_test)}")

    model_names = ["CNN-LSTM", "Encoder", "Transformer"]
    rows: list[dict[str, float | int | str]] = []

    for m in model_names:
        for run_id in range(1, int(args.runs) + 1):
            tf.keras.backend.clear_session()
            model = build_model(m, input_shape)
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
            pr_auc, opt_th, y_pred = evaluate_with_threshold(y_test, y_prob, float(args.target_precision))
            acc = float(np.mean(y_pred == y_test))

            print(f"[{m} - Run {run_id}] PR-AUC={pr_auc:.4f}, TH={opt_th:.4f}, ACC={acc:.4f}")
            print(classification_report(y_test, y_pred, target_names=["Not Bottom (0)", "Bottom (1)"], zero_division=0))

            rows.append(
                {
                    "model": m,
                    "run_id": run_id,
                    "pr_auc": pr_auc,
                    "optimal_threshold": opt_th,
                    "accuracy": acc,
                    "pred_bottom_count": int(np.sum(y_pred)),
                    "true_bottom_count": int(np.sum(y_test)),
                }
            )

    result_df = pd.DataFrame(rows)
    summary_df = (
        result_df.groupby("model", as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            th_mean=("optimal_threshold", "mean"),
            th_std=("optimal_threshold", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
        )
        .round(4)
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(f"- {out_csv}")
    print(f"- {summary_csv}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
