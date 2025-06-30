from fastapi import FastAPI
from services import  getWeatherService

app = FastAPI()  # get Fast api instance

@app.get("/weather/") # sue fast api instance to link the route with the handler
def getWeather(city_name:str):
    return getWeatherService(city_name) # return response
