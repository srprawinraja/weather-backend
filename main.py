from fastapi import FastAPI
from services import predict_temp
from database import init_db

app = FastAPI()
init_db()

@app.post("/temp/")
def predictTemp(latitude:str, longitude:str, start_date:str, end_date:str, daily:str, timezone:str, city:str, current_temperature:str):
    return predict_temp(latitude, longitude, start_date, end_date, daily, timezone, city, current_temperature)
