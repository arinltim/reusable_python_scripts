# Install dependencies:
# pip install selenium streamlink openai-whisper
# git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
import os

import google.generativeai as genai
import re
import whisper
from datetime import datetime, date
import time
from streamlink import Streamlink
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json


def realstream(page_url: str,
                timeout: float = 15.0) -> (str, str):
    """
    1) Opens the page in headless Chrome.
    2) Clicks the play overlay to start playback.
    3) Reads the <video> element’s blob src.
    4) Monitors Performance logs for a response whose mimeType indicates video/HLS.
       Returns (blob_src, real_stream_url).
    """

    # temp_dir = "media/directory/"

    # temp_dir = tempfile.mkdtemp()
    # chrome_opts = Options()
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--incognito")
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    # chrome_opts.add_argument("--headless")
    # # chrome_opts.add_argument(f"--user-data-dir={temp_dir}")
    # chrome_opts.add_argument("--disable-gpu")
    # chrome_opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)
    print("chrome opts loaded")
    driver.get(page_url)
    time.sleep(2)  # allow page and player to load

    # Click the overlay button to start playback (ads may play first)
    try:
        btns = driver.find_elements(By.CSS_SELECTOR, "button.amp-pause-overlay")
        btns[1].click()
        print("Clicked play overlay.")
    except Exception:
        print("Play overlay not found or not clickable; proceeding.")

    time.sleep(3)

    # Extract blob URL from the <video> tag
    video_el = driver.find_element(By.CSS_SELECTOR, "video.amp-media-element")
    blob_src = video_el.get_attribute("src")
    print("Blob URL:", blob_src)

    # Inspect Performance logs for the real stream URL by MIME type
    deadline = time.time() + timeout
    real_url = None
    while time.time() < deadline and not real_url:
        for entry in driver.get_log("performance"):
            msg = json.loads(entry["message"])["message"]
            if msg.get("method") == "Network.responseReceived":
                res = msg.get("params", {}).get("response", {})
                mime = res.get("mimeType", "")
                url = res.get("url", "")
                if "application/vnd.apple.mpegurl" in mime or mime.startswith("video/"):
                    real_url = url
                    break
        if not real_url:
            time.sleep(0.5)

    driver.quit()

    if not real_url:
        raise RuntimeError("Unable to detect real stream URL from performance logs.")
    print("Real stream URL detected:", real_url)
    return blob_src, real_url


def streamlink(stream_url: str,
               duration_sec,
               output_path) -> str:
    """
    Uses Streamlink to capture both audio and video from an HLS/MPEG-DASH live stream
    for `duration_sec` seconds and writes directly to a transport stream file (.ts).
    """
    session = Streamlink()
    streams = session.streams(stream_url)
    if 'best' not in streams:
        raise RuntimeError(f"No playable streams found for {stream_url}")
    stream = streams['best']

    print(f"Opening streamlink session for {stream_url}...")
    fd = stream.open()
    start = time.time()

    with open(output_path, 'wb') as outfile:
        print(f"Recording to {output_path} for {duration_sec}s...")
        while True:
            data = fd.read(8192)
            if not data:
                print("Stream ended early.")
                break
            outfile.write(data)
            if time.time() - start > duration_sec:
                print("Reached target duration.")
                break

    fd.close()
    print(f"Saved transport stream to {output_path}")
    return output_path

def transcription(path,location):
    model = whisper.load_model("small")  # You can use "small", "medium", or "large"

    # Transcribe an audio file (Replace with actual file path)
    # try:
    #     with open(path, "r", encoding="utf-8") as file:
    #         video = file.read()
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    result = model.transcribe(path)

    with open(location, 'w') as f:
        f.write(result["text"])

    return location

def gemini(location,path):
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat_session = model.start_chat(history=[])

    # context = result["text"]
    file_path = path

    # Open the file and read its content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    context = content

    current_date = datetime.now()
    day_of_week = current_date.strftime("%A")
    city = location


    prompt_context = """Extract the context given above and give the values of temperature low, temperature high, wind speed, precipitation probability and cloud conditions. 
                        Consider today as """ + day_of_week + """ and fetch the mentioned values for """+ city +""".If a range is given consider the highest. 
                        Do not give anything other than the numeric values and mention the values only if mentioned in the context, if not present give none.
                        If any of the value is not present give "0" as the value, only for sky conditions give "other".
                        """
    prompt= ("Context: "+context+'\n'+prompt_context)
    response = chat_session.send_message(prompt)
    # print(response.text)
    # print(type(response.text))
    output = str(response.text)
    patterns = {
        'Temperature-High': r'Temperature high:\s*(\S+)',
        'Temperature-Low': r'Temperature low:\s*(\S+)',
        'Wind': r'Wind speed:\s*(\S+)',
        'Precipitation': r'Precipitation probability:\s*(\S+)',
        'Sky-Conditions': r'Cloud conditions:\s*(\S+)'
    }

    # Extract values using regex
    weather_data = {key: re.search(pattern, output).group(1) if re.search(pattern, output) else None
                    for key, pattern in patterns.items()}
    # print(weather_data)
    return {
        'Temperature-High': weather_data["Temperature-High"],
        'Temperature-Low': weather_data["Temperature-Low"],
        'Wind': weather_data["Wind"],
        'Precipitation': weather_data["Precipitation"],
        'Sky-Conditions': weather_data["Sky-Conditions"]
    }

def fetch():
    print("init")
    # url = "https://www.kare11.com/watch"
    # url = "https://www.wcnc.com/watch"
    url = "https://www.wusa9.com/watch"

    print(url)
    _, real_stream = realstream(url,15)
    clip_path = streamlink(real_stream,duration_sec= 1*60,output_path= "clip.ts")
    print("Done! Clip saved at:", clip_path)

    # clip_path = "1min_clip_new.ts"
    transcript = transcription(path=clip_path, location="transcript.txt")
    print(transcript)
    print("Done! transcript saved")
    op = gemini(location="charlotte",path=transcript)
    print(op)

if __name__=='__main__':
    fetch()
