
> Drone Weather Prediction System:
 
End-to-end pipeline for drone companies to download historical weather data, train a machine learning model, and predict weather and sunrise/sunset times for any global drone mission using only drone logs.  
 Includes robust error handling, automatic timezone/country detection, and works offline for sun calculations.


🚁 Drone Weather Prediction Project — User Guide
====================================================

Project Overview
----------------
This project enables a drone company to:
- Download historical weather data for any location.
- Train a machine learning model to predict weather and sunrise/sunset.
- Predict weather and sun times for drone missions, anywhere in the world, using only drone logs.

All scripts are designed for reliability, offline use (for sun calculations), and easy troubleshooting.

-------------------------------------------------
1. import_weather.py
-------------------------------------------------
Purpose:
    Downloads historical weather data from Open-Meteo for a specified location and date range.

How to Use:
1. Edit Location and Dates:
    Open import_weather.py and set the latitude, longitude, start_date, and end_date in the params dictionary.
2. Install Requirements:
    pip install openmeteo-requests pandas requests-cache retry-requests
3. Run the Script:
    python import_weather.py
4. Output:
    - hourly_data.csv — Hourly weather data
    - daily_data.csv — Daily weather data

-------------------------------------------------
2. train_model.py
-------------------------------------------------
Purpose:
    Trains a machine learning model to predict weather and sunrise/sunset using the downloaded data.

How to Use:
1. Ensure Data Exists:
    Make sure hourly_data.csv and daily_data.csv are present.
2. Install Requirements:
    pip install pandas numpy joblib xgboost scikit-learn
3. Run the Script:
    python train_model.py
4. Output:
    - model.pkl — Trained model
    - scaler.pkl — Feature scaler
    - feature_names.pkl — List of input features

-------------------------------------------------
3. prediction_weather.py
-------------------------------------------------
Purpose:
    Predicts weather and calculates accurate sunrise/sunset for the drone’s actual location and time, using the trained model and drone log data.

How to Use:
1. Prepare Drone Log:
    - Place drone_logs.csv in the project folder.
    - Required columns: latitude, longitude, timestamp (ISO format), and ideally altitude, twist.linear.x/y/z for wind speed.
2. Install Requirements:
    pip install pandas joblib numpy astral timezonefinder reverse_geocoder pytz
3. Run the Script:
    python prediction_weather.py
4. Output:
    - predicted_weather.csv — Detailed predictions for each log entry
    - final_prediction.csv — Average/summary prediction
    - Console output with:
        - Drone’s nearest city, state, and country
        - Local and India sunrise/sunset times
        - Weather summary (rain, clouds, etc.)

-------------------------------------------------
File Descriptions
-------------------------------------------------
- hourly_data.csv / daily_data.csv:
    Weather data downloaded from Open-Meteo.
- model.pkl, scaler.pkl, feature_names.pkl:
    Artifacts from model training, used for prediction.
  - model.pkl (trained weather prediction model)
  - scaler.pkl (StandardScaler for input features)
  - feature_names.pkl (ensures feature consistency at prediction time)
  - Features used: ['temperature_2m', 'relative_humidity_2m', 'precipitation_probability', 'cloud_cover']

- drone_logs.csv:
    Drone log file (exported from your drone system).
- predicted_weather.csv:
    Weather predictions for each log entry.
- final_prediction.csv:
    Average prediction and accurate sunrise/sunset for the mission.

-------------------------------------------------
Troubleshooting
-------------------------------------------------
- Missing columns:
    The script will tell you which columns are missing from your CSV.
- Invalid coordinates:
    The script will warn you and stop if latitude/longitude are out of range.
- Timezone not found:
    The script will default to UTC and warn you.
- Reverse geocoding fails:
    The script will still run, but city/state/country may show as "Unknown".
- Any error:
    The script prints a clear message and exits safely.

-------------------------------------------------
Best Practices
-------------------------------------------------
- Keep all scripts and CSV files in the same folder.
- If you change the location or time range, rerun all scripts in order.
- If you update the drone log format, ensure required columns are present.
- If you encounter errors, read the console message for what to fix.

-------------------------------------------------
Extending the Project
-------------------------------------------------
- Add more weather features or change the ML model in train_model.py.
- Fetch official sunrise/sunset from APIs if needed, but Astral is sufficient for most use cases.
- Integrate this pipeline into a larger drone operations dashboard or automation system.

-------------------------------------------------
Quick Start Checklist
-------------------------------------------------
1. Download weather data:
    python import_weather.py
2. Train the model:
    python train_model.py
3. Place your drone log as drone_logs.csv.
4. Predict weather for your mission:
    python prediction_weather.py

-------------------------------------------------
This guide should help any user or team member run and maintain your drone weather prediction system with confidence!
