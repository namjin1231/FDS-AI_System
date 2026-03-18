import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# 모델과 Scaler 로드
model = keras.models.load_model("models/fds_model.h5")
scaler = joblib.load("models/scaler.pkl")

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

def predict_fraud(transaction_data):
    """
    거래 데이터를 받아 부정 거래 여부를 예측합니다.
    
    Parameters:
    transaction_data (dict): 거래 데이터
    
    Returns:
    dict: 예측 결과 및 확률
    """
    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame([transaction_data])
    
    # 필요한 특성 추출
    X = df[features].values
    
    # 정규화
    X_scaled = scaler.transform(X)
    
    # 예측
    probability = model.predict(X_scaled, verbose=0)[0][0]
    prediction = "비정상 거래" if probability > 0.5 else "정상 거래"
    
    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "confidence": max(probability, 1 - probability)
    }


if __name__ == "__main__":
    # 테스트 예제
    print("\n" + "="*60)
    print("FDS (사기 탐지 시스템) 예측 테스트")
    print("="*60)
    
    # 예제 1: 정상 거래
    normal_transaction = {
        "amount": 50000,
        "transaction_hour": 14,
        "transaction_type": 1,
        "merchant_risk": 0.2,
        "device_change": 0,
        "ip_change": 0,
        "location_change": 0,
        "tx_count_10m": 2
    }
    
    result1 = predict_fraud(normal_transaction)
    print(f"\n예제 1 (정상 거래):")
    print(f"  금액: {normal_transaction['amount']}")
    print(f"  예측 결과: {result1['prediction']}")
    print(f"  사기 확률: {result1['fraud_probability']:.4f}")
    print(f"  신뢰도: {result1['confidence']:.4f}")
    
    # 예제 2: 비정상 거래
    fraud_transaction = {
        "amount": 500000,
        "transaction_hour": 3,
        "transaction_type": 2,
        "merchant_risk": 0.9,
        "device_change": 1,
        "ip_change": 1,
        "location_change": 1,
        "tx_count_10m": 12
    }
    
    result2 = predict_fraud(fraud_transaction)
    print(f"\n예제 2 (비정상 거래):")
    print(f"  금액: {fraud_transaction['amount']}")
    print(f"  예측 결과: {result2['prediction']}")
    print(f"  사기 확률: {result2['fraud_probability']:.4f}")
    print(f"  신뢰도: {result2['confidence']:.4f}")
    
    print("\n" + "="*60)
