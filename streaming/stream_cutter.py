import time
import json
import os
from datetime import datetime
from streamlink import Streamlink
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import ffmpeg
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
CLIP_DURATION_SECONDS = 20
OUTPUT_FOLDER = "output_clips"

# ──────────────────────────────────────────────────────────
# 1) Extract the direct HLS stream URL from the webpage
# ──────────────────────────────────────────────────────────
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
                buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                if buttons:
                    for play_button in buttons:
                        if play_button.is_displayed() and play_button.is_enabled():
                            play_button.click()
                            logger.info(f"→ Clicked the play button using selector: {selector}")
                            clicked = True
                            break
                    if clicked:
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
                if url.endswith(".m3u8") or "application/vnd.apple.mpegurl" in mime_type:
                    hls_url = url
                    break
        if not hls_url:
            time.sleep(0.5)

    if not hls_url:
        raise RuntimeError("Failed to detect HLS URL within the timeout period.")
    return hls_url

# ──────────────────────────────────────────────────────────
# 2) Continuously cut the live video stream using FFmpeg's segment muxer
# ──────────────────────────────────────────────────────────
def live_stream_cutter(hls_url: str):
    """
    Uses a single, continuous FFmpeg process to cut the stream into
    seamless segments. This version first uses Streamlink to get the
    best quality stream, avoiding issues with data tracks like SCTE-35.
    """
    logger.info("→ Using Streamlink to find the best quality stream...")
    try:
        sess = Streamlink()
        streams = sess.streams(hls_url)
        if not streams:
            logger.error("Could not find any streams.")
            return

        stream_url = streams["best"].url
        logger.info(f"✔ Best quality stream URL found: {stream_url}")

    except Exception as e:
        logger.error(f"Failed to get stream URL with Streamlink: {e}")
        return


    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        logger.info(f"→ Created output directory: {OUTPUT_FOLDER}")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_pattern = os.path.join(OUTPUT_FOLDER, f"clip_{session_timestamp}_%03d.mp4")

    logger.info("→ Starting continuous FFmpeg segmenting process...")
    logger.info(f"→ Clips will be {CLIP_DURATION_SECONDS} seconds long.")
    logger.info(f"→ Output pattern: {output_pattern}")
    logger.info("→ Press Ctrl+C to stop the process.")

    try:
        # Define the input stream
        input_stream = ffmpeg.input(stream_url)

        # Correct way to map specific streams using ffmpeg-python
        # Select all video streams (v) and all audio streams (a)
        # This is equivalent to -map 0:v -map 0:a in FFmpeg CLI
        process = (
            ffmpeg
            .output(
                input_stream.video,
                input_stream.audio,
                output_pattern,
                c='copy',
                f='segment',
                segment_time=CLIP_DURATION_SECONDS,
                reset_timestamps=1,
                fflags='+ignidx'
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error("An FFmpeg error occurred:")
        logger.error(e.stderr.decode())
    except KeyboardInterrupt:
        logger.info("\n→ FFmpeg process interrupted by user. Stopping capture.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

# ──────────────────────────────────────────────────────────
# 3) Main function to orchestrate the process
# ──────────────────────────────────────────────────────────
def main():
    """
    Main execution function.
    """
    page_url = "https://www.wusa9.com/watch"

    driver = None

    try:
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--log-level=3")
        opts.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

        driver = webdriver.Chrome(options=opts)
        logger.info("Browser instance created.")

        hls_url = get_hls_url(driver, page_url)
        logger.info(f"✔ HLS URL found: {hls_url}")

        live_stream_cutter(hls_url)

    except RuntimeError as e:
        logger.error(f"Application Error: {e}")
    except KeyboardInterrupt:
        logger.info("\n→ Process stopped by user. Closing browser...")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        if driver:
            driver.quit()
            logger.info("→ Browser closed.")

if __name__ == "__main__":
    main()