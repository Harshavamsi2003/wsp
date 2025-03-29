import streamlit as st
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pytz
import os

# Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "845f4560291a584d511189a4a1839e28")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize geocoder
geolocator = Nominatim(user_agent="wind_prediction_app")

def get_city_coordinates(city_name):
    """Get latitude and longitude for a city"""
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def fetch_current_weather(lat, lon):
    """Get current weather data from OpenWeather API"""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'temperature': data['main']['temp'],
            'pressure': data['main']['pressure'],
            'humidity': data['main']['humidity'],
            'timestamp': datetime.now(pytz.utc)
        }
    return None

def fetch_historical_data(lat, lon, days=30):
    """Get historical weather data"""
    base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(days=days)
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        url = f"{base_url}?lat={lat}&lon={lon}&dt={timestamp}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for hour in data['hourly']:
                all_data.append({
                    'timestamp': datetime.fromtimestamp(hour['dt']),
                    'wind_speed': hour['wind_speed'],
                    'wind_direction': hour.get('wind_deg', 0),
                    'temperature': hour['temp'],
                    'pressure': hour['pressure'],
                    'humidity': hour['humidity']
                })
        current_date += timedelta(days=1)
    
    return pd.DataFrame(all_data)

def engineer_features(df):
    """Create features for ML models"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Lag features
    for lag in [1, 3, 6]:
        df[f'wind_speed_lag_{lag}h'] = df['wind_speed'].shift(lag)
    
    # Rolling features
    df['rolling_mean_6h'] = df['wind_speed'].rolling(6, min_periods=1).mean()
    df['rolling_std_6h'] = df['wind_speed'].rolling(6, min_periods=1).std()
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def train_xgboost(X_train, y_train):
    """Train and return XGBoost model"""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save trained model to disk"""
    joblib.dump(model, f"{MODEL_DIR}/{filename}.joblib")

def load_model(filename):
    """Load trained model from disk"""
    return joblib.load(f"{MODEL_DIR}/{filename}.joblib")

# Streamlit UI
st.set_page_config(
    page_title="Wind Speed Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌬️ Live Wind Speed Prediction")

tab1, tab2 = st.tabs(["Real-time Prediction", "Model Training"])

with tab1:
    st.header("Get Live Wind Prediction")
    city = st.text_input("Enter city name", "London")
    
    if st.button("Predict Wind Speed"):
        with st.spinner("Fetching weather data..."):
            lat, lon = get_city_coordinates(city)
            if not lat or not lon:
                st.error("Could not find city coordinates")
            else:
                weather_data = fetch_current_weather(lat, lon)
                if weather_data:
                    # Prepare features
                    features = pd.DataFrame([{
                        'temperature': weather_data['temperature'],
                        'pressure': weather_data['pressure'],
                        'humidity': weather_data['humidity'],
                        'wind_direction': weather_data['wind_direction'],
                        'hour': weather_data['timestamp'].hour,
                        'day_of_week': weather_data['timestamp'].weekday(),
                        'month': weather_data['timestamp'].month,
                        'wind_speed_lag_1h': weather_data['wind_speed'],
                        'wind_speed_lag_3h': weather_data['wind_speed'],
                        'rolling_mean_6h': weather_data['wind_speed'],
                        'rolling_std_6h': 0
                    }])
                    
                    # Load model and predict
                    try:
                        model = load_model("xgboost_wind_model")
                        prediction = model.predict(features)[0]
                        
                        st.success(f"Predicted wind speed for {city}: {prediction:.2f} m/s")
                        st.metric("Current Wind Speed", f"{weather_data['wind_speed']} m/s")
                        
                        # Show weather details
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Temperature", f"{weather_data['temperature']}°C")
                        col2.metric("Pressure", f"{weather_data['pressure']} hPa")
                        col3.metric("Humidity", f"{weather_data['humidity']}%")
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                else:
                    st.error("Could not fetch weather data")

with tab2:
    st.header("Train Prediction Model")
    city = st.text_input("City for training data", "London")
    days = st.slider("Days of historical data", 7, 90, 30)
    
    if st.button("Train Model"):
        with st.spinner(f"Fetching {days} days of data for {city}..."):
            lat, lon = get_city_coordinates(city)
            if not lat or not lon:
                st.error("Could not find city coordinates")
            else:
                df = fetch_historical_data(lat, lon, days)
                if not df.empty:
                    df = engineer_features(df)
                    
                    # Prepare features and target
                    features = df.drop(columns=['timestamp', 'wind_speed'])
                    target = df['wind_speed']
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, target, test_size=0.2, shuffle=False)
                    
                    # Train model
                    model = train_xgboost(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    # Evaluate
                    r2 = r2_score(y_test, predictions)
                    mae = np.mean(np.abs(y_test - predictions))
                    
                    # Save model
                    save_model(model, "xgboost_wind_model")
                    
                    # Show results
                    st.success(f"Model trained successfully! (R²: {r2:.3f}, MAE: {mae:.3f} m/s)")
                    
                    # Plot predictions
                    fig = px.line(
                        x=df['timestamp'],
                        y=df['wind_speed'],
                        title="Historical Wind Speed"
                    )
                    st.plotly_chart(fig)
                else:
                    st.error("Could not fetch historical data")
