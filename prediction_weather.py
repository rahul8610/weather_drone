# prediction_weather.py (Final Accurate Version for India & Global)

import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from math import sqrt
from astral import LocationInfo
from astral.sun import sun
import pytz
from timezonefinder import TimezoneFinder
import reverse_geocoder as rg
import sys

def safe_get(df, col, default):
    return df[col] if col in df.columns else default

try:
    # --- Load model and supporting artifacts ---
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_features = joblib.load("feature_names.pkl")
    except Exception as e:
        print(f"❌ Error loading model or scaler: {e}")
        sys.exit(1)

    # --- Load and preprocess drone log data ---
    try:
        drone_df = pd.read_csv("drone_logs.csv")
    except Exception as e:
        print(f"❌ Error loading 'drone_logs.csv': {e}")
        sys.exit(1)

    # Compute wind speed from linear twist
    try:
        if all(col in drone_df.columns for col in ['twist.linear.x', 'twist.linear.y', 'twist.linear.z']):
            drone_df['wind_speed_10m'] = drone_df[['twist.linear.x', 'twist.linear.y', 'twist.linear.z']].fillna(0).apply(
                lambda row: sqrt(row.iloc[0]**2 + row.iloc[1]**2 + row.iloc[2]**2), axis=1)
        else:
            drone_df['wind_speed_10m'] = 0
    except Exception as e:
        print(f"⚠️ Error computing wind speed: {e}")
        drone_df['wind_speed_10m'] = 0

    # Set base weather values if missing
    try:
        drone_df['temperature_2m'] = safe_get(drone_df, 'temperature_2m', 27 + 0.0065 * (safe_get(drone_df, 'altitude', 2) - 2))
        drone_df['relative_humidity_2m'] = safe_get(drone_df, 'relative_humidity_2m', 60)
        drone_df['cloud_cover'] = safe_get(drone_df, 'cloud_cover', 50)
        drone_df['precipitation_probability'] = safe_get(drone_df, 'precipitation_probability', 30)
        drone_df['surface_pressure'] = safe_get(drone_df, 'surface_pressure', 1013)
    except Exception as e:
        print(f"⚠️ Error setting default weather values: {e}")

    # Verify timestamp presence
    if 'timestamp' not in drone_df.columns or drone_df['timestamp'].dropna().empty:
        print("❌ 'timestamp' column required for sunrise/sunset calculation and must not be empty.")
        sys.exit(1)

    # Ensure all required features exist
    missing = [f for f in expected_features if f not in drone_df.columns]
    if missing:
        print(f"❌ Required input features missing after processing: {missing}")
        sys.exit(1)

    # Prepare features for prediction
    try:
        X_input = drone_df[expected_features].copy()
        X_scaled = scaler.transform(X_input)
        predictions = model.predict(X_scaled)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

    # Output predictions per row
    pred_df = pd.DataFrame(predictions, columns=[
        'predicted_rain_sum (mm)',
        'predicted_temp_max (°C)',
        'predicted_temp_min (°C)',
        'predicted_sunrise_hour',
        'predicted_sunset_hour'
    ])
    pred_df.to_csv("predicted_weather.csv", index=False)

    # Average prediction across all rows
    final_prediction = pred_df.mean().to_frame().T

    # --- Accurate Sunrise/Sunset using Astral for drone's actual location ---
    if all(col in drone_df.columns for col in ['latitude', 'longitude']):
        lat = drone_df['latitude'].dropna().iloc[0]
        lon = drone_df['longitude'].dropna().iloc[0]
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            print(f"❌ Invalid latitude/longitude: {lat}, {lon}")
            sys.exit(1)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon, lat=lat)
        if timezone_str is None:
            print("⚠️ Could not determine timezone, defaulting to UTC.")
            timezone_str = "UTC"
        # Get city/country info
        try:
            location_info = rg.search((lat, lon), mode=1)[0]
            city_name = location_info['name']
            admin1 = location_info['admin1']
            country_code = location_info['cc']
        except Exception as e:
            print(f"⚠️ Reverse geocoding failed: {e}")
            city_name = "Unknown"
            admin1 = "Unknown"
            country_code = "XX"
    else:
        print("❌ Missing 'latitude' or 'longitude' in drone logs")
        sys.exit(1)

    # --- Timestamp handling for drone's local timezone ---
    try:
        ts_valid = pd.to_datetime(drone_df['timestamp'].dropna().iloc[0], utc=True).tz_convert(timezone_str)
    except Exception as e:
        print(f"❌ Error parsing or converting timestamp: {e}")
        sys.exit(1)

    # Astral sunrise/sunset calculation
    try:
        city = LocationInfo(name=city_name, region=admin1, timezone=timezone_str, latitude=lat, longitude=lon)
        s_data = sun(city.observer, date=ts_valid.date(), tzinfo=pytz.timezone(timezone_str))
        final_prediction['predicted_sunrise (time)'] = [s_data['sunrise']]
        final_prediction['predicted_sunset (time)'] = [s_data['sunset']]
    except Exception as e:
        print(f"❌ Astral calculation failed: {e}")
        final_prediction['predicted_sunrise (time)'] = [pd.NaT]
        final_prediction['predicted_sunset (time)'] = [pd.NaT]

    final_prediction.to_csv("final_prediction.csv", index=False)
    print("\n✅ Final weather prediction saved to final_prediction.csv")
    print("📄 All detailed predictions saved to predicted_weather.csv")

    # --- Forecast Summary ---
    print("\n[Final Weather Forecast Summary]\n")
    print(f"🌍 Drone location: {city_name}, {admin1}, {country_code} ({timezone_str})")
    rain_val = final_prediction['predicted_rain_sum (mm)'].iloc[0]
    cloud = drone_df['cloud_cover'].mean()
    sunrise_local = final_prediction['predicted_sunrise (time)'].iloc[0]
    sunset_local = final_prediction['predicted_sunset (time)'].iloc[0]
    sunrise = sunrise_local.strftime('%I:%M %p') if pd.notnull(sunrise_local) else "N/A"
    sunset = sunset_local.strftime('%I:%M %p') if pd.notnull(sunset_local) else "N/A"

    # Also show sunrise/sunset in India time
    try:
        sunrise_india = sunrise_local.astimezone(pytz.timezone("Asia/Kolkata")).strftime('%I:%M %p') if pd.notnull(sunrise_local) else "N/A"
        sunset_india = sunset_local.astimezone(pytz.timezone("Asia/Kolkata")).strftime('%I:%M %p') if pd.notnull(sunset_local) else "N/A"
    except Exception:
        sunrise_india = "N/A"
        sunset_india = "N/A"

    if rain_val > 2:
        rain_status = f"☔ Moderate Rain Expected ({rain_val:.1f} mm)"
    elif rain_val > 0.5:
        rain_status = f"🌧️ Light Rain Possible ({rain_val:.1f} mm)"
    else:
        rain_status = f"☀️ Low Chance of Rain"

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
    print(f"🌅 Sunrise: {sunrise} | 🌇 Sunset: {sunset} ({timezone_str})")
    print(f"🌅 Sunrise in India time: {sunrise_india} | 🌇 Sunset in India time: {sunset_india} (Asia/Kolkata)")
    print("✅ Drone can proceed based on this forecast.")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
