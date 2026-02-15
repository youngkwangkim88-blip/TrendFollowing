import os
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Flatten, BatchNormalization, GlobalAveragePooling1D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import precision_recall_curve, auc, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_optimized_cnn_lstm(input_shape):
    # 최적화 포인트: 필터수/LSTM유닛 감소, L2 정규화, BatchNormalization 도입
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4), # 드롭아웃 강화
        Dense(1, activation='sigmoid')
    ])
    return model

def build_optimized_encoder(input_shape):
    # 최적화 포인트: Flatten 제거 및 GlobalAveragePooling 도입하여 파라미터 폭발 방지
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling1D()(x) # 핵심 최적화! 차원을 극적으로 압축
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def run_worker(model_name, imbalance_opt, run_id):
    print(f"[WORKER START] {model_name} | {imbalance_opt} | Run {run_id}")
    
    data = np.load('data/stage7/baked_dataset_h60.npz')
    X_train, y_train, w_train = data['X_train'], data['y_train'], data['w_train']
    X_val, y_val, w_val = data['X_val'], data['y_val'], data['w_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_optimized_cnn_lstm(input_shape) if model_name == 'CNN-LSTM' else build_optimized_encoder(input_shape)
    
    class_weight = None
    loss_function = 'binary_crossentropy'
    train_sw = w_train.astype(np.float32).copy()
    val_sw = w_val.astype(np.float32).copy()
    
    if imbalance_opt == 'ClassWeight':
        pos = np.sum(y_train)
        neg = len(y_train) - pos
        cw0 = 1.0
        cw1 = (neg / pos) if pos > 0 else 1.0
        train_sw = train_sw * np.where(y_train == 1, cw1, cw0).astype(np.float32)
        val_sw = val_sw * np.where(y_val == 1, cw1, cw0).astype(np.float32)
        class_weight = None
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    
    elif imbalance_opt == 'FocalLoss':
        loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.75, gamma=2.0)
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        sample_weight=train_sw,
        validation_data=(X_val, y_val, val_sw),
        epochs=150, batch_size=256,
        callbacks=[es],
        class_weight=class_weight,
        verbose=0
    )
    
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc_val = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_th = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_pred_class = (y_pred_prob >= best_th).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    
    os.makedirs('outputs/stage7_temp', exist_ok=True)
    result_file = f"outputs/stage7_temp/res_{model_name}_{imbalance_opt}_r{run_id}.csv"
    pd.DataFrame([{
        'model': model_name, 'imbalance': imbalance_opt, 'run_id': run_id,
        'pr_auc': pr_auc_val, 'optimal_th': best_th, 'accuracy': acc,
        'pred_bottom_count': np.sum(y_pred_class), 'true_bottom_count': np.sum(y_test)
    }]).to_csv(result_file, index=False)
    
    print(f"[WORKER DONE] {model_name}({imbalance_opt}) PR-AUC: {pr_auc_val:.4f}, ACC: {acc:.4f}")

def run_manager():
    models = ['CNN-LSTM', 'Encoder']
    imbalance_opts = ['ClassWeight', 'FocalLoss']
    runs = 3 
    
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
        
    print("🚀 [Stage 7 Step 2] 모든 병렬 학습(12개) 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--imbalance', type=str)
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
        run_worker(args.model, args.imbalance, args.run_id)
    else:
        run_manager()
