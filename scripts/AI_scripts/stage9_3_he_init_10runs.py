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
    # ReLU 활성화 함수에 수학적으로 최적화된 He Normal 초기화 적용!
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
               kernel_initializer='he_normal', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal'),
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False), # LSTM은 내부 게이트(tanh/sigmoid) 구조라 기본값 유지
        Dropout(0.4),
        Dense(1, activation='sigmoid') # 출력층도 sigmoid이므로 기본값 유지
    ])
    return model

def run_worker(run_id):
    lr = 0.001
    batch_size = 256
    
    print(f"[WORKER START] He-Init Verification | Run {run_id}/10")
    
    data = np.load('data/stage7/baked_dataset_h60.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_deep_cnn_he(input_shape)
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Epoch 제한 완화 및 충분한 수렴 대기 (patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=300, batch_size=batch_size, # 300으로 증가
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
    
    os.makedirs('outputs/stage9_he_init', exist_ok=True)
    result_file = f"outputs/stage9_he_init/res_he_r{run_id}.csv"
    model.save_weights(f"outputs/stage9_he_init/model_he_r{run_id}.weights.h5")
    
    pd.DataFrame([{
        'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] He-Init Run {run_id}/10 -> PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    runs = 10
    processes = []
    for r in range(1, runs + 1):
        cmd = ['python', __file__, '--worker', '--run_id', str(r)]
        cmd[0] = sys.executable
        p = subprocess.Popen(cmd)
        processes.append(p)
        
    for p in processes:
        p.wait()
        
    print("🚀 [Stage 9-3] He-Init 적용 10번 병렬 실행 완료!")

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
            except RuntimeError as e: pass
        run_worker(args.run_id)
    else:
        run_manager()
