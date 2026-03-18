import pandas as pd
import numpy as np

rows = 5000  # 더 많은 학습 데이터

# 비정상 거래 비율: 약 5%
normal_rows = int(rows * 0.95)
fraud_rows = int(rows * 0.05)

# 정상 거래 데이터
normal_data = {
    "amount": np.random.exponential(50000, normal_rows),
    "transaction_hour": np.random.randint(0, 24, normal_rows),
    "transaction_type": np.random.randint(0, 3, normal_rows),
    "merchant_risk": np.random.uniform(0, 0.5, normal_rows),
    "device_change": np.random.randint(0, 2, normal_rows),
    "ip_change": np.random.randint(0, 2, normal_rows),
    "location_change": np.random.randint(0, 2, normal_rows),
    "tx_count_10m": np.random.randint(1, 10, normal_rows),
    "is_fraud": np.zeros(normal_rows, dtype=int)
}

# 비정상 거래 데이터 (큰 금액, 높은 위험도, IP 주소 변경 등)
fraud_data = {
    "amount": np.random.exponential(200000, fraud_rows),  # 더 큰 금액
    "transaction_hour": np.random.randint(0, 24, fraud_rows),
    "transaction_type": np.random.randint(0, 3, fraud_rows),
    "merchant_risk": np.random.uniform(0.6, 1.0, fraud_rows),  # 높은 위험도
    "device_change": np.random.binomial(1, 0.8, fraud_rows),   # 장치 변경 가능성 높음
    "ip_change": np.random.binomial(1, 0.9, fraud_rows),       # IP 변경 가능성 높음
    "location_change": np.random.binomial(1, 0.8, fraud_rows),  # 위치 변경 가능성 높음
    "tx_count_10m": np.random.randint(5, 15, fraud_rows),      # 짧은 시간에 많은 거래
    "is_fraud": np.ones(fraud_rows, dtype=int)
}

# 데이터 결합
df_normal = pd.DataFrame(normal_data)
df_fraud = pd.DataFrame(fraud_data)
df = pd.concat([df_normal, df_fraud], ignore_index=True)

# 데이터 섞기
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("app/transactions_train.csv", index=False)

print(f"Training data generated: {len(df)} samples")
print(f"Fraud samples: {df['is_fraud'].sum()} ({df['is_fraud'].sum()/len(df)*100:.2f}%)")
print(f"Normal samples: {(df['is_fraud']==0).sum()} ({(df['is_fraud']==0).sum()/len(df)*100:.2f}%)")