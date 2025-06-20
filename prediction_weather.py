# adaptive_prediction_weather.py

import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from math import sqrt
from astral import LocationInfo
from astral.sun import sun
import pytz

# --- Load trained model artifacts ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("feature_names.pkl")

# --- Load raw drone logs ---
try:
    drone_df = pd.read_csv("drone_logs.csv")
except FileNotFoundError:
    raise FileNotFoundError("❌ 'drone_logs.csv' not found. Place it in the project folder.")

# --- Extract usable features from raw telemetry ---
if all(col in drone_df.columns for col in ['twist.linear.x', 'twist.linear.y', 'twist.linear.z']):
    drone_df['wind_speed_10m'] = drone_df[['twist.linear.x', 'twist.linear.y', 'twist.linear.z']].fillna(0).apply(
        lambda row: sqrt(row.iloc[0]**2 + row.iloc[1]**2 + row.iloc[2]**2), axis=1)
else:
    drone_df['wind_speed_10m'] = 0

base_temp = 27
if 'altitude' in drone_df.columns:
    drone_df['temperature_2m'] = base_temp + 0.0065 * (drone_df['altitude'].fillna(0) - 2)
else:
    drone_df['temperature_2m'] = base_temp

if 'relative_humidity_2m' not in drone_df.columns:
    drone_df['relative_humidity_2m'] = 60

if 'cloud_cover' not in drone_df.columns:
    drone_df['cloud_cover'] = 50

if 'timestamp' not in drone_df.columns:
    raise ValueError("❌ 'timestamp' column required for aligning sunrise/sunset output")

# --- Build input feature set ---
missing = [f for f in expected_features if f not in drone_df.columns]
if missing:
    raise ValueError(f"❌ Required input features missing after processing: {missing}")

X_input = drone_df[expected_features].copy()
X_scaled = scaler.transform(X_input)

# --- Make predictions ---
predictions = model.predict(X_scaled)
pred_df = pd.DataFrame(predictions, columns=[
    'predicted_precipitation_probability (%)',
    'predicted_temp_max (°C)',
    'predicted_temp_min (°C)',
    'predicted_sunrise (unix)',
    'predicted_sunset (unix)'
])

# --- Save all per-row predictions for detailed inspection ---
pred_df.to_csv("predicted_weather.csv", index=False)

# --- Aggregate to a single final prediction ---
final_prediction = pred_df.mean().to_frame().T

# --- Calculate accurate sunrise/sunset using astral ---
if all(col in drone_df.columns for col in ['latitude', 'longitude']):
    try:
        lat = drone_df['latitude'].dropna().iloc[0]
        lon = drone_df['longitude'].dropna().iloc[0]
    except:
        raise ValueError("❌ 'latitude' or 'longitude' contains no valid values.")
else:
    raise ValueError("❌ Missing 'latitude' or 'longitude' columns in drone_logs.csv")

# Use first valid timestamp to determine the date
try:
    ts_valid = pd.to_datetime(drone_df['timestamp'].dropna().iloc[0])
    if ts_valid.tzinfo is None:
        ts_valid = ts_valid.tz_localize("Asia/Kolkata")
except Exception as e:
    raise ValueError(f"❌ Invalid timestamp format in 'timestamp' column: {e}")

# Compute sunrise/sunset
try:
    city = LocationInfo(name="DroneZone", region="Earth", timezone="Asia/Kolkata", latitude=lat, longitude=lon)
    s_data = sun(city.observer, date=ts_valid.date(), tzinfo=pytz.timezone("Asia/Kolkata"))
    final_prediction['predicted_sunrise (time)'] = s_data['sunrise']
    final_prediction['predicted_sunset (time)'] = s_data['sunset']
except Exception as e:
    raise RuntimeError(f"❌ Failed to compute sunrise/sunset: {e}")

# --- Save final output ---
final_prediction.to_csv("final_prediction.csv", index=False)
print("✅ Final weather prediction saved to final_prediction.csv")
print("📄 All detailed predictions saved to predicted_weather.csv")

# --- Summary ---
print("\n📡 Final Weather Forecast Summary:\n")
rain_prob = final_prediction['predicted_precipitation_probability (%)'].iloc[0]
cloud = drone_df['cloud_cover'].mean()
sunrise = final_prediction['predicted_sunrise (time)'].iloc[0].strftime('%I:%M %p')
sunset = final_prediction['predicted_sunset (time)'].iloc[0].strftime('%I:%M %p')

rain_status = f"☔ Chance of Rain: {rain_prob:.1f}%" if rain_prob > 20 else "🌤️ Low Chance of Rain"
if cloud > 75:
    cloud_status = f"☁️ Very Cloudy ({cloud:.1f}%)"
elif cloud > 50:
    cloud_status = f"⛅ Partly Cloudy ({cloud:.1f}%)"
elif cloud > 20:
    cloud_status = f"🌤️ Mostly Clear ({cloud:.1f}%)"
else:
    cloud_status = f"☀️ Clear Skies ({cloud:.1f}%)"

print(rain_status)
print(cloud_status)
print(f"🌅 Sunrise: {sunrise} | 🌇 Sunset: {sunset}")
print("✅ Drone can proceed based on this forecast.")
