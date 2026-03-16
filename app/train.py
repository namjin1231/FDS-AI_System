import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/transactions_train.csv")

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

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

model.fit(X_scaled)

joblib.dump(model, "models/fds_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("model trained and saved")