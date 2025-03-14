from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

load_dotenv()

# ✅ Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")



engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

from sqlalchemy import Index

class WeatherModel(Base):
    __tablename__ = "weather_models"
    city = Column(String, primary_key=True, index=True) 
    model_data = Column(LargeBinary)
    __table_args__ = (Index('idx_city', 'city'),) 
 


def init_db():
    Base.metadata.create_all(engine)
