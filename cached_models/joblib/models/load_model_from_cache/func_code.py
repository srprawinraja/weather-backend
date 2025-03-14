# first line: 59
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
