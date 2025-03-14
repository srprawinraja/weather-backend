import io
import joblib
from database import WeatherModel, SessionLocal
import zlib  
model_cache = {}

def save_model_to_db(model, city_name):
    session = SessionLocal()
    
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    
    compressed_data = zlib.compress(model_bytes.getvalue())  # ✅ Always compress before storing
    
    model_entry = WeatherModel(city=city_name, model_data=compressed_data)
    session.merge(model_entry)  # Insert or update model
    session.commit()
    session.close()
    print(f"✅ Model for {city_name} saved to database (Compressed)!")



def load_model_from_db(city_name):
    session = SessionLocal()
    model_entry = session.query(WeatherModel).filter_by(city=city_name).first()
    session.close()

    if model_entry:
        try:
            # ✅ Try decompression first
            decompressed_data = zlib.decompress(model_entry.model_data)
            model_bytes = io.BytesIO(decompressed_data)
            model = joblib.load(model_bytes)
            print(f"✅ Model for {city_name} loaded from database!")
            return model
        except zlib.error:
            # ❌ Not compressed, load normally
            model_bytes = io.BytesIO(model_entry.model_data)
            model = joblib.load(model_bytes)
            print(f"⚠ Model for {city_name} was not compressed. Loaded normally.")
            return model
    else:
        print(f"⚠ No model found for {city_name}.")
        return None

def load_model_from_cache(city_name):
    if city_name in model_cache:
        print(f"✅ Model for {city_name} loaded from cache!")
        return model_cache[city_name]  # ✅ Return cached model
    
    model = load_model_from_db(city_name)
    if model:
        model_cache[city_name] = model  # 🔥 Store in cache
    return model

