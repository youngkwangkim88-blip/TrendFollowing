import os
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_curve, auc, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def run_worker(model_name, imbalance_opt, run_id):
    print(f"[WORKER START] {model_name} | {imbalance_opt} | Run {run_id}")
    
    # 1. NPZ 초고속 로드 (0.1초 컷)
    data = np.load('data/stage6/baked_dataset_h110.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # 모델 선택
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm(input_shape) if model_name == 'CNN-LSTM' else build_encoder(input_shape)
    
    # 클래스 불균형 대책 적용
    class_weight = None
    loss_function = 'binary_crossentropy'
    train_sw = w_train.astype(np.float32).copy()
    val_sw = w_val.astype(np.float32).copy()
    
    if imbalance_opt == 'ClassWeight':
        pos = np.sum(y_train)
        neg = len(y_train) - pos
        # Option A: 정답(Bottom)에 강력한 페널티 부여
        cw0 = 1.0
        cw1 = (neg / pos) if pos > 0 else 1.0
        train_sw = train_sw * np.where(y_train == 1, cw1, cw0).astype(np.float32)
        val_sw = val_sw * np.where(y_val == 1, cw1, cw0).astype(np.float32)
        class_weight = None
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    
    elif imbalance_opt == 'FocalLoss':
        # Option B: Keras Focal Loss (정답 클래스(1)에 alpha 0.75, gamma 2.0 집중)
        loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.75, gamma=2.0)
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    # 2. Early Stopping (w_val이 완벽히 적용된 val_loss 기준)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 3. 모델 학습
    model.fit(
        X_train, y_train, 
        sample_weight=train_sw,
        validation_data=(X_val, y_val, val_sw),
        epochs=150, batch_size=256,
        callbacks=[es],
        class_weight=class_weight, # Focal Loss일 땐 None이 들어감
        verbose=0
    )
    
    # 평가
    y_pred_prob = model.predict(X_test).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc_val = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_th = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_pred_class = (y_pred_prob >= best_th).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    
    # 결과 저장
    os.makedirs('outputs/stage6_temp', exist_ok=True)
    result_file = f"outputs/stage6_temp/res_{model_name}_{imbalance_opt}_r{run_id}.csv"
    pd.DataFrame([{
        'model': model_name, 'imbalance': imbalance_opt, 'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc,
        'pred_bottom_count': np.sum(y_pred_class), 'true_bottom_count': np.sum(y_test)
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] {model_name}({imbalance_opt}) PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    models = ['CNN-LSTM', 'Encoder']
    imbalance_opts = ['ClassWeight', 'FocalLoss']
    runs = 3 # 신뢰성을 위해 각 3번씩 반복 (총 12개 태스크)
    
    processes = []
    for m in models:
        for imb in imbalance_opts:
            for r in range(1, runs + 1):
                cmd = ['python', __file__, '--worker', '--model', m, '--imbalance', imb, '--run_id', str(r)]
                cmd[0] = sys.executable
                p = subprocess.Popen(cmd)
                processes.append(p)
                
    for p in processes:
        p.wait()
        
    print("🚀 [Stage 6 Step 2] 모든 병렬 학습(12개) 완료!")
    # TODO: outputs/stage6_temp 폴더의 모든 csv를 병합하는 로직 추가

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--imbalance', type=str)
    parser.add_argument('--run_id', type=int)
    args = parser.parse_args()
    
    if args.worker:
        # GPU 메모리 할당 제한 (병렬 실행 시 OOM 방지)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        run_worker(args.model, args.imbalance, args.run_id)
    else:
        run_manager()
