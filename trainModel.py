import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor
import time

def get_tempMin_model(df):
  try:
    X = df[['apparent_temperature_mean_c', 'solar_radiation_total_mj_m2', 'wind_gust_max_kmh','temperature_max_c','temperature_min_c',
            'daylight_duration_seconds','precipitation_total_mm','evapotranspiration_mm']]
      
    Y = df[['tomorrow_temperature_min_c', 'day_after_tomorrow_temperature_min_c']]

      # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

      # Train the RandomForest Regressor
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)

      # Make predictions
    y_pred = model.predict(x_test)

    mae_tomorrow = mean_absolute_error(y_test['tomorrow_temperature_min_c'], y_pred[:, 0])
    mae_day_after = mean_absolute_error(y_test['day_after_tomorrow_temperature_min_c'], y_pred[:, 1])


    return {
      "model":model,
      "maeTomorrow":mae_tomorrow,
      "mae_day_after": mae_day_after
    }
  except Exception as e:
        raise Exception(f"An error occurred: {e}")

def get_tempMax_model(df):
  try:
    X = df[['weather_code', 'temperature_mean_c', 'temperature_max_c', 'temperature_min_c',
              'apparent_temperature_mean_c', 'apparent_temperature_max_c', 'sunset_time',
              'sunshine_duration_seconds', 'wind_speed_max_kmh', 'wind_gust_max_kmh',
              'wind_direction_dominant_deg', 'evapotranspiration_mm']]
      
    Y = df[['tomorrow_temperature_max_c', 'day_after_tomorrow_temperature_max_c']]

      # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

      # Train the RandomForest Regressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(x_train, y_train)

      # Make predictions
    y_pred = model.predict(x_test)

      # Evaluate model performance
    mae_tomorrow = mean_absolute_error(y_test['tomorrow_temperature_max_c'], y_pred[:, 0])
    mae_day_after = mean_absolute_error(y_test['day_after_tomorrow_temperature_max_c'], y_pred[:, 1])


    return {
      "model":model,
      "maeTomorrow":mae_tomorrow,
      "mae_day_after": mae_day_after
    }
  except Exception as e:
        raise Exception(f"An error occurred: {e}")

def get_weather_model(df):
  try:
    X = df[['apparent_temperature_mean_c',
          'daylight_duration_seconds', 'precipitation_total_mm', 'precipitation_duration_hours',
          'wind_gust_max_kmh', 'wind_direction_dominant_deg', 'solar_radiation_total_mj_m2',
          'evapotranspiration_mm']]


    Y = df[['tomorrow_weather_code', 'weather_day_after']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x_train, y_train)

  # Make predictions
    y_pred = model.predict(x_test)

  # Print accuracy for each target
    accuracy_tomorrow = accuracy_score(y_test['tomorrow_weather_code'], y_pred[:, 0])
    accuracy_day_after = accuracy_score(y_test['weather_day_after'], y_pred[:, 1])


    return {
      "model":model,
      "accuracyTomorrow":accuracy_tomorrow*100,
      "accuracy_day_after": accuracy_day_after*100
    }
  except Exception as e:
        raise Exception(f"An error occurred: {e}")
  
def predict_temperature(model, input_features):
    # Ensure the input is in the correct shape (2D array)
  try:
      input_array = np.array(input_features).reshape(1, -1)
      predictions = model.predict(input_array)
        
      return {
            "tomorrow_pred": predictions[0][0],
            "day_after_tomorrow_pred": predictions[0][1]
        }
  except Exception as e:
    raise Exception(f"An error occurred: {e}")



