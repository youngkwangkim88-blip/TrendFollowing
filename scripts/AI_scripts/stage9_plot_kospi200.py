import os
import glob
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SpatialDropout1D
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
        if "\\u" in c:
            try:
                decoded = c.encode("utf-8").decode("unicode_escape")
                if decoded in df.columns:
                    return decoded
            except Exception:
                pass
    if required:
        raise KeyError(f"Missing column among {candidates}. columns={list(df.columns)}")
    return None


def build_deep_cnn_he(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model


def _discover_weight_files():
    patterns = [
        # latest
        'outputs/stage9_he_init/model_he_r*.weights.h5',
        'outputs/stage9_he_init/model_he_r*.h5',
        # previous versions
        #'outputs/stage9_verify/model_verify_r*.weights.h5',
        #'outputs/stage9_verify/model_verify_r*.h5',
        #'outputs/stage9_temp/model_lr*_bs*_r*.weights.h5',
        #'outputs/stage9_temp/model_lr*_bs*_r*.h5',
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return files


def analyze_and_plot(agreement_n):
    print(f"[Visualization V3] KOSPI 200 dynamic hard-voting (target votes >= {agreement_n})")

    main_matches = glob.glob('data/*kospi200*.csv')
    if not main_matches:
        print('No KOSPI200 CSV found under data/')
        return

    main_file = main_matches[0]
    df = pd.read_csv(main_file)

    date_col = _pick_col(df, ['date', 'Date', 'datetime', '\\uB0A0\\uC9DC'])
    close_col = _pick_col(df, ['close', 'C', 'Close', '\\uC885\\uAC00'])
    ticker_col = _pick_col(df, ['ticker', 'symbol', '\\uC885\\uBAA9\\uCF54\\uB4DC', '\\uD2F0\\uCEE4'], required=False)

    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df[date_col].notna()].copy()

    target_col = 'is_bottom'
    if target_col not in df.columns:
        df[target_col] = 0

    df['20d_min'] = df[close_col].rolling(window=20).min()

    features = [c for c in df.columns if c not in [date_col, ticker_col, target_col, '20d_min']]

    train_end = int(len(df) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][features])
    scaled_data = scaler.transform(df[features])

    window_size = 60
    X, dates, y_true, closes, mins_20d = [], [], [], [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:(i + window_size)])
        dates.append(df[date_col].iloc[i + window_size])
        y_true.append(df[target_col].iloc[i + window_size])
        closes.append(df[close_col].iloc[i + window_size])
        mins_20d.append(df['20d_min'].iloc[i + window_size])

    X = np.array(X)
    if len(X) == 0:
        print('Not enough rows to build sequences')
        return

    input_shape = (X.shape[1], X.shape[2])

    weight_files = _discover_weight_files()
    total_models = len(weight_files)
    if total_models == 0:
        print('No model weights found in outputs/stage9_he_init, outputs/stage9_verify, or outputs/stage9_temp/')
        return

    if agreement_n < 1:
        agreement_n = 1
    if agreement_n > total_models:
        print(f'agreement_n={agreement_n} is larger than model count={total_models}; clamping to {total_models}.')
        agreement_n = total_models

    print(f'Detected {total_models} model weights. Trigger requires >= {agreement_n} votes.')

    model = build_deep_cnn_he(input_shape)
    votes = np.zeros(len(X), dtype=np.int32)

    for w_file in weight_files:
        model.load_weights(w_file)
        preds = model.predict(X, verbose=0).ravel()
        model_threshold = np.percentile(preds, 90)
        model_vote = (preds >= model_threshold).astype(int)
        votes += model_vote

    is_pred_bottom_ensemble = (votes >= agreement_n).astype(int)

    vote_col = f'ai_votes_out_of_{total_models}'
    result_df = pd.DataFrame({
        'date': dates,
        'close': closes,
        '20d_min': mins_20d,
        'true_bottom': y_true,
        vote_col: votes,
        'ai_pred_bottom': is_pred_bottom_ensemble,
    })

    os.makedirs('outputs/analysis', exist_ok=True)
    result_df.to_csv(f'outputs/analysis/kospi200_hardvoting_N{agreement_n}_of_{total_models}.csv', index=False)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            'KOSPI 200 Close & Bottom Detection',
            'Ground Truth (is_bottom=1)',
            f'AI Votes (0~{total_models})',
        ),
    )

    fig.add_trace(go.Scatter(x=result_df['date'], y=result_df['close'], mode='lines', name='Close', line=dict(color='#2c3e50')), row=1, col=1)
    fig.add_trace(go.Scatter(x=result_df['date'], y=result_df['20d_min'], mode='lines', name='20-Day Min', line=dict(color='orange', dash='dot')), row=1, col=1)

    ai_bottoms = result_df[result_df['ai_pred_bottom'] == 1]
    fig.add_trace(
        go.Scatter(
            x=ai_bottoms['date'],
            y=ai_bottoms['close'],
            mode='markers',
            name=f'AI Pred (Votes>={agreement_n})',
            marker=dict(color='red', size=12, symbol='x', line=dict(width=2, color='DarkSlateGrey')),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=result_df['date'], y=result_df['true_bottom'], mode='lines', name='True Bottom', line=dict(color='green', shape='hv')), row=2, col=1)

    fig.add_trace(go.Bar(x=result_df['date'], y=result_df[vote_col], name='Votes', marker_color='purple'), row=3, col=1)
    fig.add_hline(y=agreement_n, line_dash='dash', line_color='red', annotation_text=f'Trigger (N={agreement_n})', row=3, col=1)

    fig.update_layout(title_text=f'KOSPI 200: Strict Hard Voting Ensemble (N={agreement_n}/{total_models})', height=1000, hovermode='x unified')
    fig.write_html(f'outputs/analysis/kospi200_hardvoting_N{agreement_n}_plot.html')
    print(f'Done: outputs/analysis/kospi200_hardvoting_N{agreement_n}_plot.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agreement_n', type=int, default=7, help='Minimum AI votes to mark bottom')
    args = parser.parse_args()
    analyze_and_plot(args.agreement_n)
