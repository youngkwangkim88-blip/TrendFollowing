#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pivot-point classifier (LightGBM).")
    p.add_argument("--data-path", default="data/kospi200_20260212.csv")
    p.add_argument("--label-path", default="outputs/pivotal point/pivot_point_labels_kospi200.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Load and merge
    df_raw = pd.read_csv(args.data_path, low_memory=False)
    df_label = pd.read_csv(args.label_path, low_memory=False)
    df = pd.merge(df_raw, df_label[["날짜", "label_20d"]], on="날짜", how="inner")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜").reset_index(drop=True)

    # 2) Data cleansing
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

    # 3) Feature engineering
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

    features = ["O", "H", "L", "C", "V", "Body_Ratio", "Upper_Tail_Ratio", "Lower_Tail_Ratio", "ATR_10"]
    target = "label_20d"
    df = df.dropna(subset=features + [target]).reset_index(drop=True)

    # 4) Sample weights
    n = len(df)
    df["sample_weight"] = 0.7 + (1.0 - 0.7) * np.linspace(0, 1, n)

    # 5) Time split
    split = int(n * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[features]
    y_train = train_df[target]
    w_train = train_df["sample_weight"]
    X_test = test_df[features]
    y_test = test_df[target]

    # 6) Train model
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # 7) Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[-1, 0, 1], target_names=["Bottom Pivot(-1)", "Normal(0)", "Top Pivot(1)"]))


if __name__ == "__main__":
    main()
