#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import argrelextrema
from sklearn.metrics import auc, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping
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
tf.get_logger().setLevel("ERROR")
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage5 Ticker-Weighted Pivot Benchmark")
    p.add_argument("--panel-csv", default="data/krx100_adj_5000.csv")
    p.add_argument("--index-csv", default="data/kospi200_20260212.csv", help="Standalone index CSV")
    p.add_argument("--symbols", default="1028,005930,000660", help="comma-separated ticker list")
    p.add_argument("--index-symbol", default="1028", help="Main index ticker")
    p.add_argument("--index-weight", type=float, default=0.7, help="Weight ratio for the index")
    p.add_argument("--seq-length", type=int, default=60)
    p.add_argument("--label-order", type=int, default=10)
    p.add_argument("--label-expand", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--target-precision", type=float, default=0.30)
    p.add_argument("--target-model", default="ALL", help="Run only one model (CNN-LSTM|Encoder|Transformer)")
    p.add_argument("--run-id", type=int, default=0, help="Run only one run id (1..N). 0 means all runs.")
    p.add_argument("--out-csv", default="outputs/pivotal point/ai_pivot_stage5_results.csv")
    p.add_argument("--summary-csv", default="outputs/pivotal point/ai_pivot_stage5_summary.csv")
    return p.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing column among {candidates}")
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
    date_col = _pick_col(df, ["날짜", "date", "Date", "?좎쭨", "일자", "?쇱옄"])
    open_col = _pick_col(df, ["O", "open", "Open", "시가", "?쒓?"])
    high_col = _pick_col(df, ["H", "high", "High", "고가", "怨좉?"])
    low_col = _pick_col(df, ["L", "low", "Low", "저가", "?媛"])
    close_col = _pick_col(df, ["C", "close", "Close", "종가", "醫낃?"])
    volume_col = _pick_col(df, ["V", "volume", "Volume", "거래량", "嫄곕옒??"], required=False)

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
        raise ValueError("Not enough rows")

    x = x.rename(columns={open_col: "O", high_col: "H", low_col: "L", close_col: "C", volume_col: "V"})
    return x


def _build_sequences(df: pd.DataFrame, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1e-8
    X_seq: list[np.ndarray] = []
    y_lab: list[int] = []

    for i in range(len(df) - seq_length):
        window = df.iloc[i : i + seq_length]
        local_min_val = float(window["L"].min())
        price_range = float(window["H"].max()) - local_min_val + epsilon

        scaled_ohlc = (window[["O", "H", "L", "C"]] - local_min_val) / price_range
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
        y_lab.append(int(df.iloc[i + seq_length - 1]["Target_Bottom"]))

    return np.array(X_seq, dtype=np.float32), np.array(y_lab, dtype=np.int32)


def load_multi_ticker_weighted_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _parse_symbols(args.symbols)
    index_symbol = _normalize_symbol(args.index_symbol)
    if index_symbol not in symbols:
        symbols.append(index_symbol)

    ticker_train_arrays: list[tuple[str, np.ndarray, np.ndarray]] = []
    X_test_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []

    symbols_to_process = list(symbols)
    if index_symbol in symbols_to_process:
        symbols_to_process.remove(index_symbol)

    if args.index_csv and Path(args.index_csv).exists():
        print(f"[INFO] loading main index CSV: {args.index_csv}")
        idx_df = pd.read_csv(args.index_csv, low_memory=False)
        try:
            prepared = _prepare_symbol_frame(idx_df, args.seq_length, args.label_order, args.label_expand)
            X, y = _build_sequences(prepared, args.seq_length)
            if len(X) >= 100:
                split = int(len(X) * 0.8)
                ticker_train_arrays.append((index_symbol, X[:split], y[:split]))
                X_test_list.append(X[split:])
                y_test_list.append(y[split:])
                print(f"[OK] {index_symbol} (Main Index): train={split}, test={len(X)-split}")
            else:
                print(f"[SKIP] {index_symbol}: sequence data too small")
        except Exception as e:
            print(f"[ERROR] {index_symbol}: failed to load index CSV ({e})")
    else:
        print(f"[WARNING] index CSV not found ({args.index_csv}). trying panel lookup for {index_symbol}.")
        symbols_to_process.append(index_symbol)

    raw = pd.read_csv(args.panel_csv, dtype={"ticker": str}, low_memory=False)
    ticker_col = _pick_col(raw, ["ticker", "종목코드", "symbol", "醫낅ぉ肄붾뱶"])
    raw[ticker_col] = raw[ticker_col].astype(str).map(_normalize_symbol)

    for sym in symbols_to_process:
        g = raw[raw[ticker_col] == sym].copy()
        if g.empty:
            print(f"[SKIP] {sym}: no data in panel")
            continue
        try:
            prepared = _prepare_symbol_frame(g, args.seq_length, args.label_order, args.label_expand)
            X, y = _build_sequences(prepared, args.seq_length)
            if len(X) < 100:
                print(f"[SKIP] {sym}: sequence data too small")
                continue
            split = int(len(X) * 0.8)
            ticker_train_arrays.append((sym, X[:split], y[:split]))
            X_test_list.append(X[split:])
            y_test_list.append(y[split:])
            print(f"[OK] {sym}: train={split}, test={len(X)-split}")
        except Exception as e:
            print(f"[SKIP] {sym}: {e}")

    if not ticker_train_arrays or not X_test_list:
        raise SystemExit("No valid datasets were built. Check --panel-csv / --index-csv / --symbols.")

    y_train_all = np.concatenate([ytr for _, _, ytr in ticker_train_arrays])
    total_samples = len(y_train_all)
    count_1 = int(np.sum(y_train_all))
    count_0 = total_samples - count_1

    base_w0 = total_samples / (2.0 * count_0) if count_0 > 0 else 1.0
    base_w1 = (total_samples / (2.0 * count_1)) ** 0.5 if count_1 > 0 else 1.0

    total_base_weight = float(np.sum(np.where(y_train_all == 1, base_w1, base_w0)))
    other_syms = [s for s, _, _ in ticker_train_arrays if s != index_symbol]
    n_others = len(other_syms)

    X_train_final: list[np.ndarray] = []
    y_train_final: list[np.ndarray] = []
    w_train_final: list[np.ndarray] = []

    for sym, xtr, ytr in ticker_train_arrays:
        w_base = np.where(ytr == 1, base_w1, base_w0).astype(np.float32)
        s_base = float(np.sum(w_base))

        if sym == index_symbol:
            target_sum = total_base_weight * float(args.index_weight)
        else:
            target_sum = total_base_weight * (1.0 - float(args.index_weight)) / n_others if n_others > 0 else 0.0

        multiplier = target_sum / s_base if s_base > 0 else 1.0
        w_final = w_base * multiplier

        X_train_final.append(xtr)
        y_train_final.append(ytr)
        w_train_final.append(w_final)

    X_train = np.concatenate(X_train_final, axis=0)
    y_train = np.concatenate(y_train_final, axis=0)
    w_train = np.concatenate(w_train_final, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    w_train = w_train[idx]

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

    print("Loading multi-ticker weighted dataset...")
    X_train, X_test, y_train, y_test, w_train = load_multi_ticker_weighted_dataset(args)

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Train={X_train.shape}, Test={X_test.shape}, Bottom in test={int(np.sum(y_test))}/{len(y_test)}")

    model_names = ["CNN-LSTM", "Encoder", "Transformer"]
    if args.target_model != "ALL":
        if args.target_model not in model_names:
            raise SystemExit(f"Invalid --target-model: {args.target_model}. Choose one of {model_names} or ALL.")
        model_names = [args.target_model]
    rows: list[dict[str, float | int | str]] = []

    for m in model_names:
        run_range = range(1, int(args.runs) + 1)
        if int(args.run_id) != 0:
            run_range = [int(args.run_id)]

        for run_id in run_range:
            tf.keras.backend.clear_session()
            model = build_model(m, input_shape)

            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
            print(f"[TRAIN] {m} - Run {run_id} (max epochs={args.epochs})")
            model.fit(
                X_train,
                y_train,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                validation_split=0.2,
                sample_weight=w_train,
                callbacks=[early_stop],
                verbose=0,
            )

            y_prob = model.predict(X_test, verbose=0).flatten()
            pr_auc, opt_th, y_pred = evaluate_with_threshold(y_test, y_prob, float(args.target_precision))
            acc = float(np.mean(y_pred == y_test))
            print(f"[DONE] {m} - Run {run_id}: PR-AUC={pr_auc:.4f}, TH={opt_th:.4f}, ACC={acc:.4f}")

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

    if args.target_model != "ALL":
        temp_dir = Path("outputs/pivotal point/stage5_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_csv = temp_dir / f"result_{args.target_model}_{int(args.run_id)}.csv"
        result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved stage5 shard: {out_csv}")
        return

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

    print("\nStage5 Summary")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
