import io
import joblib
from database import WeatherModel, SessionLocal

def save_model_to_db(model, city_name):
    session = SessionLocal()
    
    # Convert model to binary
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    
    # Save to PostgreSQL
    model_entry = WeatherModel(city=city_name, model_data=model_bytes.getvalue())
    session.merge(model_entry)  # Insert or update model
    session.commit()
    session.close()
    print(f"✅ Model for {city_name} saved to database!")


def load_model_from_db(city_name):
    """Load a trained model from PostgreSQL."""
    session = SessionLocal()
    model_entry = session.query(WeatherModel).filter_by(city=city_name).first()
    session.close()

    if model_entry:
        model_bytes = io.BytesIO(model_entry.model_data)
        model = joblib.load(model_bytes)
        print(f"✅ Model for {city_name} loaded from database!")
        return model
    else:
        print(f"⚠ No model found for {city_name}.")
        return None
