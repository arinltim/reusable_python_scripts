import time
import json
import tempfile
import os
from datetime import datetime
from streamlink import Streamlink
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Configuration: Adjust these values to sync subtitles
# ──────────────────────────────────────────────────────────

# This new value reduces the delay from audio buffering, making the script more responsive.
# Whisper performs well with 3-second chunks.
CHUNK_DURATION_SECONDS = 3.0

# This delay is adjusted to make transcripts appear about 0.5 seconds BEFORE the audio is heard.
# You were experiencing a 1s delay, so we need to speed up the process by 1.5s total.
# 1s was reduced by changing the chunk duration, and the remaining 0.5s is adjusted here.
# Old value was 2.0. New value is 1.5. (2.0 - 1.5 = 0.5s faster from this variable).
# Total speed up = 1.0s (from chunk) + 0.5s (from this delay) = 1.5s.
SUBTITLE_ADJUSTMENT_SECONDS = 1.5


# ──────────────────────────────────────────────────────────
# 1) Extract the direct HLS stream URL from the webpage
# ──────────────────────────────────────────────────────────
# Pass the driver instance to this function
def get_hls_url(driver: webdriver.Chrome, page_url: str, timeout: float = 15.0) -> str:
    """
    Uses an existing Chrome browser instance to find the HLS stream URL (.m3u8).
    """
    logger.info(f"→ Opening page: {page_url} in browser...")
    driver.get(page_url)
    time.sleep(5)

    try:
        play_button_selectors = [
            "button.amp-pause-overlay"
        ]

        clicked = False
        for selector in play_button_selectors:
            try:
                play_button = driver.find_elements(By.CSS_SELECTOR, selector)[1]
                if play_button.is_displayed() and play_button.is_enabled():
                    play_button.click()
                    logger.info(f"→ Clicked the play button using selector: {selector}")
                    clicked = True
                    break
            except Exception:
                pass

        if not clicked:
            logger.info("→ No specific play button found or needed, continuing.")

    except Exception as e:
        logger.warning(f"→ An unexpected error occurred while looking for play button: {e}. Continuing.")
        pass

    time.sleep(3)

    deadline = time.time() + timeout
    hls_url = None
    logger.info("→ Monitoring network traffic for HLS playlist (.m3u8)...")
    while time.time() < deadline and not hls_url:
        for entry in driver.get_log("performance"):
            message = json.loads(entry["message"])["message"]
            if message.get("method") == "Network.responseReceived":
                response = message["params"]["response"]
                mime_type = response.get("mimeType", "")
                url = response.get("url", "")
                if "application/vnd.apple.mpegurl" in mime_type or url.endswith(".m3u8"):
                    hls_url = url
                    break
        if not hls_url:
            time.sleep(0.5)

    if not hls_url:
        raise RuntimeError("Failed to detect HLS URL within the timeout period.")
    return hls_url

# ──────────────────────────────────────────────────────────
# 2) Continuously transcribe the live audio stream
# ──────────────────────────────────────────────────────────
def live_transcriber(stream_url: str):
    """
    Opens an audio stream and continuously transcribes it, applying a
    delay to synchronize with the player.
    """
    logger.info("→ Loading Whisper model ('small')...")
    model = whisper.load_model("small")
    logger.info("→ Model loaded.")

    logger.info("→ Opening audio stream via Streamlink...")
    sess = Streamlink()

    try:
        streams = sess.streams(stream_url)
        audio_stream = streams.get('bestaudio') or streams.get('audio') or streams['best']
    except Exception as e:
        raise RuntimeError(f"Could not retrieve streams or find a suitable audio stream: {e}")

    if not audio_stream:
        raise RuntimeError("Could not find a suitable audio stream after attempting all options.")

    fd = audio_stream.open()
    logger.info("→ Audio stream opened. Starting live transcription...")

    with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        while True:
            buffer = b""
            start_time = time.time()
            while time.time() - start_time < CHUNK_DURATION_SECONDS:
                data = fd.read(8192)
                if not data:
                    logger.info("→ End of audio stream detected.")
                    return
                buffer += data

            with open(temp_file_name, "wb") as f:
                f.write(buffer)

            try:
                result = model.transcribe(temp_file_name, language="en", fp16=False)
                text = result["text"].strip()
                if text:
                    # This delay is the primary tool for synchronization.
                    time.sleep(SUBTITLE_ADJUSTMENT_SECONDS) # Sync delay
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] {text}") # Keep print for immediate user feedback
            except Exception as e:
                logger.error(f"An error occurred during transcription: {e}")
    finally:
        logger.info("→ Cleaning up temporary file and closing audio stream.")
        fd.close()
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)


# ──────────────────────────────────────────────────────────
# 3) Main function to orchestrate the process
# ──────────────────────────────────────────────────────────
def main():
    """
    Main execution function.
    """
    page_url = "https://www.kare11.com/watch"

    driver = None # Initialize driver to None

    try:
        # Initialize Chrome options
        opts = Options()
        # Comment out the line below to run in a visible browser window for debugging
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--log-level=3") # Suppress console noise from browser
        opts.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

        # Create the WebDriver instance here in main()
        driver = webdriver.Chrome(options=opts)
        logger.info("Browser instance created.")

        hls_url = get_hls_url(driver, page_url) # Pass the driver instance
        driver.quit()
        logger.info(f"✔ HLS URL found: {hls_url}")

        live_transcriber(hls_url)

    except RuntimeError as e:
        logger.error(f"Application Error: {e}")
    except KeyboardInterrupt:
        logger.info("\n→ Process stopped by user. Closing browser...")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        # Ensure the browser is closed only when the main function truly finishes
        if driver:
            driver.quit()
            logger.info("→ Browser closed.")

if __name__ == "__main__":
    main()