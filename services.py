
import requests
import pandas as pd
from io import StringIO
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder 
from model import predict_temperature, get_tempMax_model, get_tempMin_model, get_weather_model
from features import maxFeatures, minFeatures, weatherFeatures

BASE_URL_HISTORICAL = 'https://archive-api.open-meteo.com/v1/archive?'
BASE_URL_CURRENT = 'https://api.open-meteo.com/v1/forecast?'
BASE_URL_LAT_LONG = 'http://api.openweathermap.org/geo/1.0/direct?'
load_dotenv()
API_KEY = os.getenv("API_KEY")

def getWeatherService(city_name:str):
    try:
        latLongdata = get_lat_long(city_name)

        if latLongdata is None:
            raise ValueError("invalid city name")
        lat = latLongdata["lat"]
        lon = latLongdata["lon"]

        obj = TimezoneFinder() 
        timeZone = obj.timezone_at(lng=lon, lat=lat)
        previousDayDate = get_previous_day()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
        csv_file_path = os.path.join(BASE_DIR, "weatherDetail.csv")
        df = retrieve_csv_from_directory(city_name)
        if df is None:
            timeout = False
            historicalUrl = f"{BASE_URL_HISTORICAL}latitude={lat}&longitude={lon}&start_date=2000-01-01&end_date={previousDayDate}&daily=weather_code,temperature_2m_mean,temperature_2m_max,temperature_2m_min,apparent_temperature_mean,apparent_temperature_max,sunrise,daylight_duration,sunshine_duration,precipitation_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&timezone={timeZone}&format=csv"
            try:
                response = requests.get(historicalUrl, timeout=30)
            except requests.exceptions.Timeout:
                timeout=True
                print(f"The request timed out")
            if not timeout and response.status_code == 200: 
                print("historical data request worked successfully") 
                csv_file_path = StringIO(response.text)
            else:
                latLongdata = get_lat_long("madurai")
                lat = latLongdata["lat"]
                lon = latLongdata["lon"]
                timeZone = obj.timezone_at(lng=lon, lat=lat)
            
            weatherUrl = f"{BASE_URL_CURRENT}latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,uv_index_clear_sky_max,rain_sum,showers_sum,snowfall_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&hourly=temperature_2m,weather_code&current=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover,pressure_msl&forecast_days=3&timezone={timeZone}"
            
            df = read_csv(csv_file_path)
            df = prepare_regression_data(df)
            save_csv_to_directory(df, city_name)
        else:
            print("retrieved from directory")

        weatherUrl = f"{BASE_URL_CURRENT}latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,uv_index_clear_sky_max,rain_sum,showers_sum,snowfall_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration&hourly=temperature_2m,weather_code&current=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover,pressure_msl&forecast_days=3&timezone={timeZone}"
        
        #weatherUrl = 'https://2k7gl.wiremockapi.cloud/openMeteo'

        response = requests.get(weatherUrl)
        data = response.json()  
        currentWeatherDetail = getCurrentWeatherDetail(data)
        todayForecastDetails = getTodayForecastDetails(data)
        daysForecast = getDaysForecastDetail(data)
        
        tempMaxModelDetail = get_tempMax_model(df)
        tempMinModelDetail = get_tempMin_model(df)
        weatherModelDetail = get_weather_model(df)

        tempMinFeatureValue = getFeatureValueList(data["daily"], minFeatures)
        weatherFeatureValue = getFeatureValueList(data["daily"], weatherFeatures)
        tempMaxFeatureValue = getFeatureValueList(data["daily"], maxFeatures)

        tempMinModel = tempMinModelDetail["model"]
        tempMaxModel = tempMaxModelDetail["model"]
        weatherModel = weatherModelDetail["model"]

        minPredict = predict_temperature(tempMinModel, tempMinFeatureValue)
        weatherPredict = predict_temperature(weatherModel, weatherFeatureValue)
        maxPredict = predict_temperature(tempMaxModel, tempMaxFeatureValue)

        day1 = get_next_day(daysForecast[len(daysForecast)-1]["day"])
        day2 = get_next_day(day1)

        daysForecast.append({
                "day": day1,
                "img": get_weather_icon(weatherPredict["tomorrow_pred"]),
                "tempMin": int(minPredict["tomorrow_pred"]),
                "tempMax": int(maxPredict["tomorrow_pred"]),
                "weather": get_weather_description(weatherPredict["tomorrow_pred"])
        })
        daysForecast.append({
                "day": day2,
                "img": get_weather_icon(weatherPredict["day_after_tomorrow_pred"]),
                "tempMin": int(minPredict["day_after_tomorrow_pred"]),
                "tempMax": int(maxPredict["day_after_tomorrow_pred"]),
                "weather": get_weather_description(weatherPredict["day_after_tomorrow_pred"])
        })
        return {
            "statusCode": 200,
            "city": city_name,
            "weather": currentWeatherDetail["weather"],
            "img": currentWeatherDetail["img"],
            "temp": currentWeatherDetail["temp"],
            "todayForecast":todayForecastDetails,
            "dayForecast":daysForecast,
            "airCondition":[
            {
                "label1":"wind speed",
                "value1": currentWeatherDetail["windSpeed"],
                "label2":"wind direction",
                "value2": currentWeatherDetail["windDirection"]
            },
            {
                "label1": "wind Gusts",
                "value1": currentWeatherDetail["windGusts"][0],
                "label2": "pressure",
                "value2": currentWeatherDetail["pressure"]
            }
        ],
        "modelPerformance":[
            {
            "maeForTodayMaxTemp":tempMaxModelDetail["maeTomorrow"],
            "maeForTomorrowMaxTemp": tempMaxModelDetail["mae_day_after"]
            },
            {
            "maeForTodayMinTemp":tempMinModelDetail["maeTomorrow"],
            "maeForTomorrowMinTemp": tempMinModelDetail["mae_day_after"]
            },
            {
            "accuracyForTodayMinTemp":weatherModelDetail["accuracyTomorrow"],
            "accuracyForTomorrowMinTemp": weatherModelDetail["accuracy_day_after"]
            }
        ]          
}
        

    except ValueError as e:
        return {
            "statusCode": 400,
            "error": "Bad Request",
            "message": str(e)
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "error": "Bad Request",
            "message": str(e)
        }

def read_csv(file_path):
  df = pd.read_csv(file_path, skiprows=3)
  df.columns = [
    "timestamp",
    "weather_code",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "apparent_temperature_mean_c",
    "apparent_temperature_max_c",
    "sunset_time",
    "daylight_duration_seconds",
    "sunshine_duration_seconds",
    "precipitation_total_mm",
    "precipitation_duration_hours",
    "wind_speed_max_kmh",
    "wind_gust_max_kmh",
    "wind_direction_dominant_deg",
    "solar_radiation_total_mj_m2",
    "evapotranspiration_mm"]
  df.drop(columns=["timestamp"], inplace=True)

  df['sunset_time'] = pd.to_datetime(df['sunset_time'], errors='coerce', format='%Y-%m-%dT%H:%M')
  df['sunset_time'] = df['sunset_time'].dt.hour * 3600 + df['sunset_time'].dt.minute * 60 + df['sunset_time'].dt.second
  df["weather_code"] = df["weather_code"].apply(lambda x: classify_weather_code(x))


  df.dropna()
  return df


def prepare_regression_data(df):
  df["tomorrow_weather_code"] = df["weather_code"].shift(-1)
  df['weather_day_after'] = df['weather_code'].shift(-2)
  df["tomorrow_temperature_max_c"] = df["temperature_max_c"].shift(-1)
  df["day_after_tomorrow_temperature_max_c"] = df["temperature_max_c"].shift(-2)
  df["tomorrow_temperature_min_c"] = df["temperature_min_c"].shift(-1)
  df["day_after_tomorrow_temperature_min_c"] = df["temperature_min_c"].shift(-2)

  df = df.dropna(subset=["tomorrow_weather_code", 'weather_day_after', 'tomorrow_temperature_max_c', "day_after_tomorrow_temperature_max_c", 'tomorrow_temperature_min_c', 'day_after_tomorrow_temperature_min_c'])
  return df

def getTodayForecastDetails(data):
    try:
        hourlyData = data["hourly"]
        todayForecastDetails = []
        for i in range(0, 25):
            time = hourlyData["time"][i]
            time_part = time.split("T")[1]  # Extract only "00:00"
            temp = hourlyData["temperature_2m"][i]
            img = get_weather_icon(hourlyData["weather_code"][i])
            normalTime = railway_to_normal(time_part)
            todayForecastDetails.append(
                {
                    "time": normalTime,
                    "img": img,
                    "temp": temp
                }
            )
        return todayForecastDetails
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def getDaysForecastDetail(data):
    try:
        dailyData = data["daily"]
        dailysData=[]
        for i in range(0, len(dailyData["time"])):
            day = get_day_name(dailyData["time"][i])
            img = get_weather_icon(dailyData["weather_code"][i])
            weather = get_weather_description(dailyData["weather_code"][i])
            minTemp = dailyData["temperature_2m_min"][i]
            maxTemp = dailyData["temperature_2m_max"][i]
            dailysData.append({
                "day": day,
                "img": img,
                "tempMin": minTemp,
                "tempMax": maxTemp,
                "weather": weather
            })

        return dailysData
    except Exception as e:
        raise Exception(f"An error occurred: {e}")



def getCurrentWeatherDetail(data):
    try:
        current = data["current"]
        temp = current["temperature_2m"]
        img = get_weather_icon(current["weather_code"])
        weather = get_weather_description(current["weather_code"])
        windSpeed = current["wind_speed_10m"]
        windDirection = current["wind_direction_10m"]
        windGusts = current["wind_gusts_10m"],
        pressure = current["pressure_msl"]
        return {
            "weather": weather,
            "img": img,
            "temp": temp,
            "windSpeed": windSpeed,
            "windDirection": windDirection,
            "windGusts": windGusts,
            "pressure": pressure
        }
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def get_lat_long(city_name):
    
    url = f"{BASE_URL_LAT_LONG}q={city_name}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()  
    if len(data)==0:
        return None
    else:
        return{
            "lat": data[0]["lat"],
            "lon": data[0]["lon"]
        }

def get_weather_icon(weather_code):
    weather_map = {
        0: "https://openweathermap.org/img/wn/01d@2x.png",
        **dict.fromkeys([1, 2, 3], "https://openweathermap.org/img/wn/02d@2x.png"),
        **dict.fromkeys([45, 48], "https://openweathermap.org/img/wn/50d@2x.png"),
        **dict.fromkeys([51, 53, 55, 56, 57], "https://openweathermap.org/img/wn/09d@2x.png"),
        **dict.fromkeys([61, 63, 65, 66, 67], "https://openweathermap.org/img/wn/11d@2x.png"),
        **dict.fromkeys([71, 73, 75, 77], "https://openweathermap.org/img/wn/13d@2x.png"),
        **dict.fromkeys([80, 81, 82, 85, 86, 95, 96, 99], "https://openweathermap.org/img/wn/11d@2x.png"),
    }
    return weather_map.get(weather_code)

def get_weather_description(code):
    weather_dict = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Drizzle: Light intensity",
        53: "Drizzle: Moderate intensity",
        55: "Drizzle: Dense intensity",
        56: "Freezing Drizzle: Light intensity",
        57: "Freezing Drizzle: Dense intensity",
        61: "Rain: Slight intensity",
        63: "Rain: Moderate intensity",
        65: "Rain: Heavy intensity",
        66: "Freezing Rain: Light intensity",
        67: "Freezing Rain: Heavy intensity",
        71: "Snow fall: Slight intensity",
        73: "Snow fall: Moderate intensity",
        75: "Snow fall: Heavy intensity",
        77: "Snow grains",
        80: "Rain showers: Slight intensity",
        81: "Rain showers: Moderate intensity",
        82: "Rain showers: Violent intensity",
        85: "Snow showers: Slight intensity",
        86: "Snow showers: Heavy intensity",
        95: "Thunderstorm: Slight or moderate",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    
    return weather_dict.get(code)

def get_day_name(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")

def getFeatureValueList(data, features):
    values = []

    for feature in features:
        if "mean" in feature:
            max_key = feature.replace("mean", "max")
            min_key = feature.replace("mean", "min")

            max_value = data.get(max_key, [None])[-1]
            min_value = data.get(min_key, [None])[-1]

            if max_value is not None and min_value is not None:
                values.append((max_value + min_value) / 2)
            else:
                print(f"Warning: Missing data for {feature}")
                values.append(None)
        else:
            value = data.get(feature, [None])[-1]
            if feature == "sunset" or feature == "sunrise":
                value=time_to_seconds(value)
            values.append(value)

    return values


def railway_to_normal(time_str):
    try:
        # Convert from 24-hour format to 12-hour format
        normal_time = datetime.strptime(time_str, "%H:%M").strftime("%I:%M %p")
        return normal_time.lstrip("0")  # Remove leading zero for better formatting
    except ValueError:
        return "Invalid time format"
    
def time_to_seconds(timestamp_str):
    time_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M")
    return time_obj.hour * 3600 + time_obj.minute * 60
    
def get_next_day(day):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day = day.capitalize()  # Ensure first letter is uppercase
    if day in weekdays:
        next_day_index = (weekdays.index(day) + 1) % 7  # Get the next day's index
        return weekdays[next_day_index]
    else:
        raise ValueError("Invalid day name. Please provide a valid weekday.")

def get_previous_day():
    return (datetime.now() - timedelta(days=1)).date()

def retrieve_csv_from_directory(city_name):
    # Define the directory and the file path
    directory_path = 'weather_data'  # Same as the directory used during saving
    csv_file_path = os.path.join(directory_path, f"{city_name}_weather_data.csv")
    
    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"CSV for {city_name} retrieved successfully.")
        return df
    else:
        print(f"❌ File for {city_name} not found.")
        return None
    
def save_csv_to_directory(csv_content, city_name):
    # Ensure the directory exists
    directory_path = 'weather_data'  # You can change this to any path you prefer
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Create the file path using the city name
    csv_file_path = os.path.join(directory_path, f"{city_name}_weather_data.csv")
    
    # Save the CSV content to the file
    csv_content.to_csv(csv_file_path, index=False)
    print(f"CSV for {city_name} saved to {csv_file_path}")

def classify_weather_code(code):
    if code == 0:
        return 0  # Clear
    elif code in [1, 2, 3, 45, 48]:
        return 1  # Cloudy
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82, 95, 96, 99]:
        return 2  # Rainy
    elif code in [71, 73, 75, 77, 85, 86]:
        return 3  # Snowy
    else:
        return np.nan  # Unknown
