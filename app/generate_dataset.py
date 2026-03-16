import pandas as pd
import numpy as np

rows = 1000

data = {
    "amount": np.random.exponential(50000, rows),
    "transaction_hour": np.random.randint(0,24,rows),
    "transaction_type": np.random.randint(0,3,rows),
    "merchant_risk": np.random.uniform(0,1,rows),
    "device_change": np.random.randint(0,2,rows),
    "ip_change": np.random.randint(0,2,rows),
    "location_change": np.random.randint(0,2,rows),
    "tx_count_10m": np.random.randint(1,10,rows)
}

df = pd.DataFrame(data)

df.to_csv("data/transactions_train.csv", index=False)

print("training data generated")