from fastapi import FastAPI
from services import  getWeatherService

app = FastAPI()

@app.get("/weather/")
def getWeather(city_name:str):
    return getWeatherService(city_name)
