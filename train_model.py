# train_model.py

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Load data
hourly_df = pd.read_csv("hourly_data.csv")
daily_df = pd.read_csv("daily_data.csv")
hourly_df['date'] = pd.to_datetime(hourly_df['date'])
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Merge hourly and daily on date
hourly_df['day'] = hourly_df['date'].dt.date
daily_df['day'] = daily_df['date'].dt.date
merged_df = pd.merge(hourly_df, daily_df, on='day', how='inner')

# Define features and targets
features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'cloud_cover']
targets = ['rain', 'temperature_2m_max', 'temperature_2m_min', 'sunrise', 'sunset']
expected_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'cloud_cover']
# Ensure all expected features are present 

# Filter and drop missing
merged_df = merged_df[features + targets].dropna()

# Prepare inputs and outputs
X = merged_df[features].copy()
y = merged_df[targets].copy()

# Fix types
X.columns = [str(col) for col in X.columns]  # ensure consistent feature names
y['sunrise'] = y['sunrise'].astype(int)
y['sunset'] = y['sunset'].astype(int)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature order
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
base_model = xgb.XGBRegressor()
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model.pkl")

print("✅ Model training complete.")
print("📦 Saved files:")
print("  - model.pkl (trained weather prediction model)")
print("  - scaler.pkl (StandardScaler for input features)")
print("  - feature_names.pkl (ensures feature consistency at prediction time)")
print(f"📊 Features used: {expected_features}")
print(f"🧠 Model type: {type(model).__name__}")
print("✅ All artifacts are ready for use in prediction_weather.py.")
