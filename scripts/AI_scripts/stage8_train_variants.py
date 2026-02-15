import os
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import precision_recall_curve, auc, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 🧠 4가지 아키텍처 Variants 정의
# ==========================================

def build_base_wide(input_shape):
    """1. Base Wide: Stage 5 복원 + SpatialDropout(CNN 과적합 방지)"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        SpatialDropout1D(0.2), # CNN 특징맵 일부를 통째로 꺼서 과적합 방지
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_heavy_wide(input_shape):
    """2. Heavy Wide: 1.5배 체급 (필터 96, LSTM 80) + 약한 L2 정규화"""
    model = Sequential([
        Conv1D(filters=96, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(80, return_sequences=False, kernel_regularizer=regularizers.l2(1e-4)), # 약한 브레이크
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_deep_cnn(input_shape):
    """3. Deep CNN: 종으로 깊게 (다층 CNN)"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'), # CNN 2단 콤보
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_deep_lstm(input_shape):
    """4. Deep LSTM: 시계열 기억력을 깊게 (다층 LSTM)"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        SpatialDropout1D(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True), # 다음 LSTM에게 시퀀스를 넘김
        LSTM(32, return_sequences=False), # LSTM 2단 콤보
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

# ==========================================
# ⚙️ 병렬 워커 및 매니저 로직
# ==========================================

def run_worker(variant_name, run_id):
    print(f"[WORKER START] {variant_name} | Run {run_id}")
    
    # 1. 60일치(Stage7) 구워진 데이터 0.1초 컷 로딩
    data = np.load('data/stage7/baked_dataset_h60.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 모델 선택
    if variant_name == 'Base_Wide': model = build_base_wide(input_shape)
    elif variant_name == 'Heavy_Wide': model = build_heavy_wide(input_shape)
    elif variant_name == 'Deep_CNN': model = build_deep_cnn(input_shape)
    elif variant_name == 'Deep_LSTM': model = build_deep_lstm(input_shape)
    else: raise ValueError("Unknown variant")
    
    # 컴파일 (클래스 가중치 없이 순수 아키텍처 성능 테스트)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early Stopping
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 학습
    model.fit(
        X_train, y_train, 
        sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=150, batch_size=256,
        callbacks=[es],
        verbose=0
    )
    
    # 평가
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc_val = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_th = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_pred_class = (y_pred_prob >= best_th).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    
    # 저장
    os.makedirs('outputs/stage8_temp', exist_ok=True)
    result_file = f"outputs/stage8_temp/res_{variant_name}_r{run_id}.csv"
    pd.DataFrame([{
        'model_variant': variant_name, 'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc,
        'pred_bottom_count': np.sum(y_pred_class), 'true_bottom_count': np.sum(y_test)
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] {variant_name} PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    variants = ['Base_Wide', 'Heavy_Wide', 'Deep_CNN', 'Deep_LSTM']
    runs = 3 # 각 3번씩 총 12개 병렬 실행
    
    processes = []
    for v in variants:
        for r in range(1, runs + 1):
            cmd = ['python', __file__, '--worker', '--variant', v, '--run_id', str(r)]
            cmd[0] = sys.executable
            p = subprocess.Popen(cmd)
            processes.append(p)
            
    for p in processes:
        p.wait()
        
    print("🚀 [Stage 8] 모든 아키텍처 Variant 병렬 테스트(12개) 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--variant', type=str)
    parser.add_argument('--run_id', type=int)
    args = parser.parse_args()
    
    if args.worker:
        # GPU OOM 방지
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                pass
        run_worker(args.variant, args.run_id)
    else:
        run_manager()
