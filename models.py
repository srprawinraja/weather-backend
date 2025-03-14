import io
import joblib
import lz4.frame
import numpy as np
import threading
from functools import lru_cache
from joblib import Memory
from database import WeatherModel, SessionLocal

# Persistent memory cache
memory = Memory(location="./cached_models", verbose=0)

# Store frequently accessed models in RAM
model_cache = {}

def save_model_to_db(model, city_name):
    with SessionLocal() as session:
        model_bytes = io.BytesIO()
        
        # 🔥 Use NumPy fast serialization
        np.save(model_bytes, model, allow_pickle=True)
        
        # 🔥 Optimized LZ4 compression
        compressed_data = lz4.frame.compress(
            model_bytes.getbuffer(), 
            compression_level=3, 
            store_size=False, 
            block_linked=1
        )
        
        model_entry = WeatherModel(city=city_name, model_data=compressed_data)
        session.merge(model_entry)
        session.commit()
    
    print(f"✅ Model for {city_name} saved to database (Ultra-Fast LZ4)!")

def load_model_from_db(city_name):
    with SessionLocal() as session:
        model_entry = session.query(WeatherModel).filter_by(city=city_name).first()

    if model_entry:
        try:
            # 🔥 Optimized LZ4 decompression
            decompressed_data = lz4.frame.decompress(model_entry.model_data)
            
            # 🔥 NumPy fast loading
            model = np.load(io.BytesIO(decompressed_data), allow_pickle=True)
            print(f"✅ Model for {city_name} loaded from database (Ultra-Fast LZ4)!")
            return model
        except lz4.frame.LZ4FrameError:  # ✅ Fixed exception
            model_bytes = io.BytesIO(model_entry.model_data)
            model = joblib.load(model_bytes)
            print(f"⚠ Model for {city_name} was not compressed. Loaded normally.")
            return model
    else:
        print(f"⚠ No model found for {city_name}.")
        return None

@lru_cache(maxsize=20)
@memory.cache
def load_model_from_cache(city_name):
    if city_name in model_cache:
        print(f"✅ Model for {city_name} loaded from cache!")
        return model_cache[city_name]  

    model = load_model_from_db(city_name)
    if model:
        model_cache[city_name] = model  
    return model

def async_load_model(city_name):
    thread = threading.Thread(target=load_model_from_cache, args=(city_name,))
    thread.start()
