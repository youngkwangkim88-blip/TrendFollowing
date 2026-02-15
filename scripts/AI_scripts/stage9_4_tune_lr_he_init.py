import os
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_recall_curve, auc, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def run_worker(lr, run_id):
    batch_size = 256 # 가장 안정적이었던 배치 사이즈 고정
    
    print(f"[WORKER START] LR: {lr} | Run {run_id}/5")
    
    data = np.load('data/stage7/baked_dataset_h60.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_deep_cnn_he(input_shape)
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # 충분한 수렴을 보장하는 콜백
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=300, batch_size=batch_size,
        callbacks=[es, reduce_lr],
        verbose=0
    )
    
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc_val = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_th = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_pred_class = (y_pred_prob >= best_th).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    
    # 결과 저장
    os.makedirs('outputs/stage9_lr_tune_v2', exist_ok=True)
    result_file = f"outputs/stage9_lr_tune_v2/res_lr{lr}_r{run_id}.csv"
    model.save_weights(f"outputs/stage9_lr_tune_v2/model_lr{lr}_r{run_id}.weights.h5")
    
    pd.DataFrame([{
        'learning_rate': lr, 'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] LR: {lr} | Run {run_id}/5 -> PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    learning_rates = [0.0013, 0.001, 0.00075] # 3가지 LR 탐색
    runs_per_lr = 5 # 각각 5번씩 실행 (총 15개 병렬 프로세스)
    
    processes = []
    for lr in learning_rates:
        for r in range(1, runs_per_lr + 1):
            cmd = [sys.executable, __file__, '--worker', '--lr', str(lr), '--run_id', str(r)]
            p = subprocess.Popen(cmd)
            processes.append(p)
            
    for p in processes:
        p.wait()
        
    print("🚀 [Stage 9-4] He-Init 최적 학습률 병렬 탐색 (15개) 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--run_id', type=int)
    args = parser.parse_args()
    
    if args.worker:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e: pass
        run_worker(args.lr, args.run_id)
    else:
        run_manager()
