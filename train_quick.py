import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("1. Loading train.csv...")
df = pd.read_csv('train.csv')

# Use a small sample so it trains lightning fast
df = df.sample(n=20000, random_state=42).copy()

print("2. Engineering Features...")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

X = df[['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend']]
y = df['sales']

print("3. Training the AI...")
model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
model.fit(X, y)

print("4. Saving the Brain...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_model.pkl')

print("SUCCESS: models/rf_model.pkl has been generated!")