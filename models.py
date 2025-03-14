import io
import joblib
import lz4.frame
from functools import lru_cache
from database import WeatherModel, SessionLocal

model_cache = {}

# ✅ Faster compression using LZ4
def save_model_to_db(model, city_name):
    session = SessionLocal()
    
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes, compress=3, protocol=4)

    compressed_data = lz4.frame.compress(model_bytes.getvalue())  # 🔥 Faster compression
    
    model_entry = WeatherModel(city=city_name, model_data=compressed_data)
    session.merge(model_entry)
    session.commit()
    session.close()
    print(f"✅ Model for {city_name} saved to database (LZ4 Compressed)!")

# ✅ Faster decompression using LZ4
def load_model_from_db(city_name):
    session = SessionLocal()
    model_entry = session.query(WeatherModel).filter_by(city=city_name).first()
    session.close()

    if model_entry:
        try:
            decompressed_data = lz4.frame.decompress(model_entry.model_data)  # 🔥 Faster decompression
            model_bytes = io.BytesIO(decompressed_data)
            model = joblib.load(model_bytes)
            print(f"✅ Model for {city_name} loaded from database (LZ4)!") 
            return model
        except lz4.frame.LZ4FError:
            model_bytes = io.BytesIO(model_entry.model_data)
            model = joblib.load(model_bytes)
            print(f"⚠ Model for {city_name} was not compressed. Loaded normally.")
            return model
    else:
        print(f"⚠ No model found for {city_name}.")
        return None

# ✅ Keep models in memory for instant access
@lru_cache(maxsize=5)
def load_model_from_cache(city_name):
    if city_name in model_cache:
        print(f"✅ Model for {city_name} loaded from cache!")
        return model_cache[city_name]  # ✅ Return cached model
    
    model = load_model_from_db(city_name)
    if model:
        model_cache[city_name] = model  # 🔥 Store in cache
    return model
