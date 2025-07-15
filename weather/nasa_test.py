import requests
from datetime import datetime

def get_weather_data(latitude, longitude, date):
    # Convert date to required formats
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    year = date_obj.year
    doy = date_obj.timetuple().tm_yday  # Day of year
    date_str = date.replace('-', '')

    # API endpoint and parameters
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": date_str,
        "end": date_str,
        "latitude": latitude,
        "longitude": longitude,
        "community": "AG",  # Use 'RE' for renewable energy, 'AG' for agriculture
        "parameters": "T2M_MAX,T2M_MIN,WS2M,PRECTOT,CLOUD_AMT",
        "format": "JSON",
        "user": "anonymous"
    }

    # Make API request
    response = requests.get(url, params=params, verify=False)
    data = response.json()
    # print(data)

    # Extract data using day-of-year key (e.g., "2020167")
    # doy_key = f"{year}{str(doy).zfill(3)}"
    doy_key = date_str

    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_high": data['properties']['parameter']['T2M_MAX'][doy_key],
        "temperature_low": data['properties']['parameter']['T2M_MIN'][doy_key],
        "wind_speed": data['properties']['parameter']['WS2M'][doy_key],
        "precipitation": data['properties']['parameter']['PRECTOTCORR'][doy_key],
        "cloud_cover": data['properties']['parameter']['CLOUD_AMT'][doy_key]
    }

# Example usage
# Charlotte, NC
target_latitude = 35.23
target_longitude = -80.84
data = get_weather_data(target_latitude, target_longitude, "2025-06-01")
print(data)