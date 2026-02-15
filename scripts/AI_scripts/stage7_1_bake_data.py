import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 60
MAIN_INDEX = "001028"
MAIN_WEIGHT = 0.70
TARGET_COL = "is_bottom"


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column among {candidates}. columns={list(df.columns)}")
    return None


def _normalize_symbol(sym: object) -> str:
    s = str(sym).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s


def _ensure_target(df: pd.DataFrame, low_col: str, order: int = 10, expand: int = 1) -> pd.DataFrame:
    out = df.copy()
    if TARGET_COL in out.columns:
        out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).astype(int)
        return out
    out[TARGET_COL] = 0
    mins = argrelextrema(out[low_col].values, np.less_equal, order=order)[0]
    last = len(out) - 1
    for idx in mins:
        lo = max(0, int(idx) - int(expand))
        hi = min(last, int(idx) + int(expand))
        out.loc[lo:hi, TARGET_COL] = 1
    return out


def create_sequences(data: np.ndarray, labels: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)


def process_ticker(df: pd.DataFrame, features: list[str], weight: float):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    val_scaled = scaler.transform(val_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_tr, y_tr = create_sequences(train_scaled, train_df[TARGET_COL].values, WINDOW_SIZE)
    X_v, y_v = create_sequences(val_scaled, val_df[TARGET_COL].values, WINDOW_SIZE)
    X_te, y_te = create_sequences(test_scaled, test_df[TARGET_COL].values, WINDOW_SIZE)

    w_tr = np.full(len(y_tr), weight, dtype=np.float32)
    w_v = np.full(len(y_v), weight, dtype=np.float32)
    return (X_tr, y_tr, w_tr), (X_v, y_v, w_v), (X_te, y_te)


def bake_data():
    print(f"[Stage7 Step1] Data baking start (window={WINDOW_SIZE})")

    panel_file = "data/krx100_adj_5000.csv"
    index_file = "data/kospi200_20260212.csv"
    if not os.path.exists(panel_file):
        raise FileNotFoundError(panel_file)
    if not os.path.exists(index_file):
        raise FileNotFoundError(index_file)

    panel = pd.read_csv(panel_file, dtype={"ticker": str}, low_memory=False)
    idx_df = pd.read_csv(index_file, dtype={"ticker": str}, low_memory=False)

    panel_ticker_col = _pick_col(panel, ["ticker", "종목코드", "symbol", "티커"])
    panel_date_col = _pick_col(panel, ["date", "날짜", "Date"])
    idx_ticker_col = _pick_col(idx_df, ["ticker", "종목코드", "symbol", "티커"])
    idx_date_col = _pick_col(idx_df, ["date", "날짜", "Date"])

    for d, dc, tc in [(panel, panel_date_col, panel_ticker_col), (idx_df, idx_date_col, idx_ticker_col)]:
        d[dc] = pd.to_datetime(d[dc], errors="coerce")
        d[tc] = d[tc].map(_normalize_symbol)
        d.dropna(subset=[dc], inplace=True)

    panel_cols = {
        "o": _pick_col(panel, ["O", "open", "Open"]),
        "h": _pick_col(panel, ["H", "high", "High"]),
        "l": _pick_col(panel, ["L", "low", "Low"]),
        "c": _pick_col(panel, ["C", "close", "Close"]),
        "v": _pick_col(panel, ["V", "volume", "Volume"], required=False),
    }
    if panel_cols["v"] is None:
        panel["V"] = 0.0
        panel_cols["v"] = "V"

    idx_cols = {
        "o": _pick_col(idx_df, ["O", "open", "Open"]),
        "h": _pick_col(idx_df, ["H", "high", "High"]),
        "l": _pick_col(idx_df, ["L", "low", "Low"]),
        "c": _pick_col(idx_df, ["C", "close", "Close"]),
        "v": _pick_col(idx_df, ["V", "volume", "Volume"], required=False),
    }
    if idx_cols["v"] is None:
        idx_df["V"] = 0.0
        idx_cols["v"] = "V"

    X_train_list, y_train_list, w_train_list = [], [], []
    X_val_list, y_val_list, w_val_list = [], [], []
    X_test_list, y_test_list = [], []

    tickers = sorted(panel[panel_ticker_col].dropna().unique().tolist())
    sub_tickers = [t for t in tickers if t != MAIN_INDEX]
    sub_weight = 0.30 / len(sub_tickers) if sub_tickers else 0.0

    main = idx_df[idx_df[idx_ticker_col] == MAIN_INDEX].copy()
    if main.empty:
        main = idx_df.copy()
    main = main.sort_values(idx_date_col).reset_index(drop=True)
    main = main.rename(
        columns={
            idx_cols["o"]: "O",
            idx_cols["h"]: "H",
            idx_cols["l"]: "L",
            idx_cols["c"]: "C",
            idx_cols["v"]: "V",
        }
    )
    main = _ensure_target(main, "L")
    features = ["O", "H", "L", "C", "V"]
    (X_tr, y_tr, w_tr), (X_v, y_v, w_v), (X_te, y_te) = process_ticker(main, features, MAIN_WEIGHT)
    X_train_list.append(X_tr); y_train_list.append(y_tr); w_train_list.append(w_tr)
    X_val_list.append(X_v); y_val_list.append(y_v); w_val_list.append(w_v)
    X_test_list.append(X_te); y_test_list.append(y_te)

    for sym in sub_tickers:
        g = panel[panel[panel_ticker_col] == sym].copy()
        g = g.sort_values(panel_date_col).reset_index(drop=True)
        g = g.rename(
            columns={
                panel_cols["o"]: "O",
                panel_cols["h"]: "H",
                panel_cols["l"]: "L",
                panel_cols["c"]: "C",
                panel_cols["v"]: "V",
            }
        )
        g = _ensure_target(g, "L")
        if len(g) < WINDOW_SIZE + 50:
            continue
        try:
            (X_tr, y_tr, w_tr), (X_v, y_v, w_v), (X_te, y_te) = process_ticker(g, features, sub_weight)
            if len(X_tr) == 0 or len(X_v) == 0 or len(X_te) == 0:
                continue
            X_train_list.append(X_tr); y_train_list.append(y_tr); w_train_list.append(w_tr)
            X_val_list.append(X_v); y_val_list.append(y_v); w_val_list.append(w_v)
            X_test_list.append(X_te); y_test_list.append(y_te)
        except Exception:
            continue

    if not X_train_list or not X_val_list or not X_test_list:
        raise SystemExit("No valid sequences generated. Check input CSV schema/data length.")

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    w_train = np.concatenate(w_train_list)
    X_val = np.concatenate(X_val_list)
    y_val = np.concatenate(y_val_list)
    w_val = np.concatenate(w_val_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train, w_train = X_train[idx], y_train[idx], w_train[idx]

    os.makedirs("data/stage7", exist_ok=True)
    np.savez_compressed(
        "data/stage7/baked_dataset_h60.npz",
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, w_val=w_val,
        X_test=X_test, y_test=y_test,
    )

    print(f"[Data Baked] Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
    print("Saved: data/stage7/baked_dataset_h60.npz")


if __name__ == "__main__":
    bake_data()
