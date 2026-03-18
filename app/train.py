import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

# 데이터 로드
df = pd.read_csv("app/transactions_train.csv")

features = [
    "amount",
    "transaction_hour",
    "transaction_type",
    "merchant_risk",
    "device_change",
    "ip_change",
    "location_change",
    "tx_count_10m"
]

X = df[features].values
y = df["is_fraud"].values

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련/검증 데이터 분할 (80:20)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# MLP 신경망 모델 구축
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(len(features),)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # 이진 분류 (정상/비정상)
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

# 모델 요약 출력
print("\n" + "="*50)
print("MLP 신경망 모델 구조")
print("="*50)
model.summary()

# Early Stopping 콜백 정의
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 모델 훈련
print("\n" + "="*50)
print("모델 훈련 시작...")
print("="*50)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 모델 평가
print("\n" + "="*50)
print("검증 데이터셋 평가 결과")
print("="*50)
val_loss, val_accuracy, val_auc = model.evaluate(X_val, y_val, verbose=0)
print(f"검증 손실(Loss): {val_loss:.45f}")
print(f"검증 정확도(Accuracy): {val_accuracy:.45f}")
print(f"검증 AUC: {val_auc:.45f}")

# 모델과 Scaler 저장
os.makedirs("models", exist_ok=True)
model.save("models/fds_model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("\n" + "="*50)
print("모델 저장 완료")
print("="*50)
print(f"모델 파일: models/fds_model.h5")
print(f"Scaler 파일: models/scaler.pkl")