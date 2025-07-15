import json
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Path to JSON mapping lat,long to weather URLs
target_json = 'tegna_markets.json'
uv_index_chart = {
    "low": 2,
    "moderate": 5,
    "high": 8,
    "very high": 10,
    "extreme": 11
}

latlong = "32.78,-96.8"

# Configure headless Chrome
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

# Instantiate WebDriver (ensure chromedriver is in PATH)
driver = webdriver.Chrome(options=options)

# Load the mapping
with open(target_json) as f:
    url_map = json.load(f)

url = url_map[latlong]
results = {}
try:
    driver.get(url)
    # Wait for page to load
    time.sleep(3)

    # Example scraping: adjust selectors as needed
    # Extract current temperature
    temp_elem = driver.find_elements(By.CSS_SELECTOR, '.weather-10-day__temperature-high')[1]
    temperature = temp_elem.text

    # Extract wind speed
    wind_elem = driver.find_elements(By.CSS_SELECTOR, '.weather-10-day__wind-number')[1]
    wind = wind_elem.text

    # Extract precipitation probability
    precip_elem = driver.find_elements(By.CSS_SELECTOR, '.weather-10-day__precipitation-number')[1]
    precipitation = precip_elem.text

    trigger = driver.find_elements(By.CSS_SELECTOR, '.weather-10-day__expander')[1]
    trigger.click()
    wait = WebDriverWait(driver, 10)
    container = wait.until(EC.visibility_of_element_located(
        (By.CSS_SELECTOR, '.weather-10-day__content')
    ))

    # Extract humidity
    humidity_elem = container.find_element(By.CSS_SELECTOR, '.weather-10-day__humidity')
    full_humidity = humidity_elem.text
    humidity = full_humidity.replace("Humidity:", "").strip()

    # Extract uv-index
    uv_elem = container.find_element(By.CSS_SELECTOR, '.weather-10-day__uv-index')
    full_uv = uv_elem.text
    uv_text = full_uv.replace("UV Index:", "").strip()
    uv = uv_index_chart[uv_text.lower()]

    # Extract pressure
    pressure = 0

    results[latlong] = {
        'Temperature': temperature,
        'Wind': wind,
        'Precipitation': precipitation,
        'Humidity': humidity,
        'Pressure': pressure,
        'UV-Index': uv
    }
    print(results)
except Exception as e:
    print(f"Error scraping {latlong} at {url}: {e}")
    raise e

# Close the browser
driver.quit()

