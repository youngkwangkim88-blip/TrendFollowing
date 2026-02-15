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

def build_deep_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def run_worker(run_id):
    # 고정된 최적 하이퍼파라미터
    lr = 0.001
    batch_size = 256
    
    print(f"[WORKER START] 10-Fold Verification | Run {run_id}/10")
    
    data = np.load('data/stage7/baked_dataset_h60.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_deep_cnn(input_shape)
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # [추가된 과적합 방지 기법] 학습이 정체되면 보폭(LR)을 줄여 미세 조정
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=150, batch_size=batch_size,
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
    
    os.makedirs('outputs/stage9_verify', exist_ok=True)
    result_file = f"outputs/stage9_verify/res_verify_r{run_id}.csv"
    weight_file = f"outputs/stage9_verify/model_verify_r{run_id}.weights.h5"
    
    model.save_weights(weight_file)
    
    pd.DataFrame([{
        'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc,
        'pred_bottom_count': np.sum(y_pred_class), 'true_bottom_count': np.sum(y_test),
        'weight_file': weight_file
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] Run {run_id}/10 완료 -> PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    runs = 10 # 10번 반복 검증
    processes = []
    
    for r in range(1, runs + 1):
        cmd = ['python', __file__, '--worker', '--run_id', str(r)]
        cmd[0] = sys.executable
        p = subprocess.Popen(cmd)
        processes.append(p)
        
    for p in processes:
        p.wait()
        
    print("🚀 [Stage 9-2] 10번의 반복 검증(Verification) 병렬 실행 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--run_id', type=int)
    args = parser.parse_args()
    
    if args.worker:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                pass
        run_worker(args.run_id)
    else:
        run_manager()
