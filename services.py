
import requests
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.ensemble import RandomForestRegressor # fixed line
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from models import save_model_to_db, load_model_from_db

BASE_URL_HISTORICAL = 'https://archive-api.open-meteo.com/v1/archive?'

def predict_temp(latitude: str, longitude: str, start_date: str, end_date: str, daily: str, timezone: str, city:str, current_temperature:str):
    url = f"{BASE_URL_HISTORICAL}latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily={daily}&timezone={timezone}&format=csv"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {
                "status_code": 503,
                "error": "Could not fetch the CSV from Open-Meteo."
            }
        
        if not response.text:
            return {
                "status_code": 400,
                "error": "The response does not contain any data."
            }
        
        file = StringIO(response.text)
        data = read_csv(file)
        
        X_temp, y_temp = prepare_regression_data(data, "temp")
        if len(X_temp) < 2:
            return {
                "status_code": 400,
                "error": "Not enough data for training."
            }
        
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
        temp_model = load_model(city)
        if(temp_model is None):
            temp_model = train_regression_model(X_train, y_train)
            save_model(temp_model, city)
        y_pred = temp_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        predicted_temp = temp_model.predict(np.array([[current_temperature]])) 
        rounded_temp = round(predicted_temp[0], 1)
        return {
            "status_code": 201,
            "predicted_temperature": f"{rounded_temp}°C",
            "mean_squared_error": mse,
            "r_squared": r2
        }

    except requests.exceptions.RequestException as e:
        return {
            "status_code": 503,
            "error": f"External API request failed: {str(e)}"
        }
    
    except Exception as e:
        return {
            "status_code": 500,
            "error": f"Internal server error: {str(e)}"
        }




def read_csv(file):
    df = pd.read_csv(file, skiprows=3)
    df.columns = ["time", "temp"]
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=200, random_state=42)
  model.fit(X, y)
  return model

def save_model(model, city_name):
    save_model_to_db(model, city_name)

def load_model(city_name):
    return load_model_from_db(city_name)