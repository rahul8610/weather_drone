# train_model.py (Final Polished Version - Clean Output, No Warnings)

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress only specific known warnings
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")

# Load and preprocess data
print("\U0001F4E5 Loading historical data...")
hourly_df = pd.read_csv("hourly_data.csv")
daily_df = pd.read_csv("daily_data.csv")

hourly_df['date'] = pd.to_datetime(hourly_df['date'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
hourly_df['day'] = hourly_df['date'].dt.date
daily_df['day'] = daily_df['date'].dt.date

merged_df = pd.merge(hourly_df, daily_df, on='day', how='inner')

# Define features and targets (only available columns)
features = [
    'temperature_2m',
    'relative_humidity_2m',
    'precipitation_probability',
    'cloud_cover',
    'wind_speed_10m',
    'surface_pressure'
]
targets = [
    'rain_sum',
    'temperature_2m_max',
    'temperature_2m_min',
    'sunrise',
    'sunset'
]

# Fill missing feature values with median or 0 if entirely NaN
for col in features:
    if col in merged_df.columns:
        if merged_df[col].isna().all():
            merged_df[col] = 0
        else:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Drop rows with missing target values
merged_df = merged_df.dropna(subset=targets)

# Convert sunrise/sunset to float hours
merged_df['sunrise_hour'] = pd.to_datetime(merged_df['sunrise'], unit='s', origin='unix').dt.hour + pd.to_datetime(merged_df['sunrise'], unit='s', origin='unix').dt.minute / 60
merged_df['sunset_hour'] = pd.to_datetime(merged_df['sunset'], unit='s', origin='unix').dt.hour + pd.to_datetime(merged_df['sunset'], unit='s', origin='unix').dt.minute / 60

# Final features and targets
X = merged_df[features]
y = merged_df[['rain_sum', 'temperature_2m_max', 'temperature_2m_min', 'sunrise_hour', 'sunset_hour']]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save artifacts
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_names.pkl")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42
))
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Output summary
print("\n✅ Model training complete.")
print("📦 Saved files:")
print("  - model.pkl (trained weather prediction model)")
print("  - scaler.pkl (StandardScaler for input features)")
print("  - feature_names.pkl (ensures feature consistency at prediction time)")
print(f"📊 Features used: {features[:4]}")
print(f"🧠 Model type: {type(model).__name__}")
print("✅ All artifacts are ready for use in prediction script.")
