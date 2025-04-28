import unittest
from services import classify_weather_code, get_day_name, railway_to_normal, time_to_seconds
import numpy as np

class TestWeatherUtils(unittest.TestCase):

    def test_classify_weather_code(self):
        self.assertEqual(classify_weather_code(0), 0)  # Clear
        self.assertEqual(classify_weather_code(2), 1)  # Cloudy
        self.assertEqual(classify_weather_code(63), 2) # Rainy
        self.assertEqual(classify_weather_code(75), 3) # Snowy
        self.assertTrue(np.isnan(classify_weather_code(999)))  # Unknown

    def test_get_day_name(self):
        self.assertEqual(get_day_name("2024-05-05"), "Sunday")
        self.assertEqual(get_day_name("2025-01-01"), "Wednesday")

    def test_railway_to_normal(self):
        self.assertEqual(railway_to_normal("00:00"), "12:00 AM")
        self.assertEqual(railway_to_normal("13:45"), "1:45 PM")
        self.assertEqual(railway_to_normal("07:30"), "7:30 AM")
        self.assertEqual(railway_to_normal("25:00"), "Invalid time format")  # Invalid input

    def test_time_to_seconds(self):
        self.assertEqual(time_to_seconds("2025-04-16T14:30"), 14*3600 + 30*60)
        self.assertEqual(time_to_seconds("2025-04-16T00:00"), 0)
        self.assertEqual(time_to_seconds("2025-04-16T23:59"), 23*3600 + 59*60)

unittest.main()
