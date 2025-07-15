import google.generativeai as genai
import re
import whisper
from datetime import datetime
import time
from streamlink import Streamlink
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
import os
import threading
import queue
import shutil

# --- CONFIGURATION ---
CLIP_DURATION_SECONDS = 20
OUTPUT_CLIPS_DIR = "output_clips"
OUTPUT_TRANSCRIPTS_DIR = "output_transcripts"
# New directories for categorized clips
CATEGORIZED_CLIPS_DIR = "categorized_clips"
WEATHER_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Weather")
BREAKING_NEWS_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Breaking_News")
GENERAL_NEWS_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "General_News")
OTHER_EVENTS_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Other_Events") # For future expansion if needed

NUM_WORKER_THREADS = 3
RUNTIME_MINUTES = 5 # Configure how long the entire process runs (e.g., 5 minutes)
# --- END CONFIGURATION ---

# Global stop event to signal all threads to terminate
stop_event = threading.Event()
# Queue to hold paths of clips ready for processing
clip_queue = queue.Queue()

# Global variable to hold the Whisper model, loaded once
WHISPER_MODEL = None

def realstream(page_url: str, timeout: float = 15.0) -> str:
    """
    Opens the page in headless Chrome, clicks play, and extracts the real HLS stream URL.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--incognito")
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = webdriver.Chrome(options=options)
    print("Browser instance created.")
    print(f"→ Opening page: {page_url} in browser...")
    driver.get(page_url)
    time.sleep(2)

    try:
        play_button_selector = "button.amp-pause-overlay"
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, play_button_selector))
        )
        btns = driver.find_elements(By.CSS_SELECTOR, play_button_selector)
        if btns:
            btns[0].click()
            print(f"→ Clicked the play button using selector: {play_button_selector}")
        else:
            print("→ Play button not found, assuming playback starts automatically or already playing.")
    except Exception as e:
        print(f"→ Play button not found or not clickable: {e}; proceeding.")

    time.sleep(3)

    print("→ Monitoring network traffic for HLS playlist (.m3u8)...")
    deadline = time.time() + timeout
    real_url = None
    while time.time() < deadline and not real_url:
        for entry in driver.get_log("performance"):
            msg = json.loads(entry["message"])["message"]
            if msg.get("method") == "Network.responseReceived":
                res = msg.get("params", {}).get("response", {})
                mime = res.get("mimeType", "")
                url = res.get("url", "")
                if "application/vnd.apple.mpegurl" in mime or url.endswith(".m3u8"):
                    real_url = url
                    break
        if not real_url:
            time.sleep(0.5)

    driver.quit()

    if not real_url:
        raise RuntimeError("Unable to detect real stream URL from performance logs.")
    print(f"✔ HLS URL found: {real_url}")
    return real_url


def transcription(model, clip_path: str, transcript_location: str, clip_filename: str) -> str:
    """
    Transcribes a video file directly using the Whisper model.
    Whisper will internally handle audio extraction.
    """
    print(f"[{threading.current_thread().name}] → Transcribing video from: {clip_path}...")
    try:
        result = model.transcribe(clip_path)
        text = result["text"]

        with open(transcript_location, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[{threading.current_thread().name}] ✔ Transcription saved to: {transcript_location}")
        return text
    except Exception as e:
        print(f"[{threading.current_thread().name}] ❗ Error transcribing {clip_filename} with Whisper: {e}")
        return ""


def gemini_analysis(text_content: str, clip_filename: str) -> str:
    """
    Analyzes the transcribed text for weather or breaking news using Gemini.
    Returns the raw Gemini output.
    """
    genai.configure(api_key="AIzaSyA2YXUaYM3lwc0msb08ANZ-a8FbvpV--PE") # REPLACE WITH YOUR ACTUAL API KEY
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt_breaking_news = f"""
    Analyze the following transcript from a news clip. Your task is to determine if the clip primarily discusses:
    1. **Weather-related content:** This includes forecasts, current weather conditions, storm warnings, climate discussions, or anything directly related to meteorological events.
    2. **Breaking News:** This refers to significant, urgent, and unfolding events that are likely to disrupt normal programming, such as major accidents, sudden political developments, natural disasters (earthquakes, tsunamis, unexpected large fires), or significant crimes.
    3. **General News:** If it's none of the above, or covers typical daily news that isn't urgent or weather-focused.

    Provide your answer as one of the following categories: 'Weather', 'Breaking News', or 'General News'.
    If it's Weather or Breaking News, also provide a concise one-sentence summary of the specific event.

    Transcript for clip {clip_filename}:
    "{text_content}"

    Example Output for Weather:
    Category: Weather
    Summary: A severe thunderstorm warning has been issued for the metropolitan area, with strong winds and heavy rainfall expected.

    Example Output for Breaking News:
    Category: Breaking News
    Summary: Police are on the scene of a major traffic accident on the I-95, advising commuters to seek alternative routes.

    Example Output for General News:
    Category: General News
    Summary: None.
    """

    try:
        response = model.generate_content(prompt_breaking_news)
        response_text = response.text.strip()
        print(f"[{threading.current_thread().name}] --- Gemini Analysis for {clip_filename} ---")
        print(f"[{threading.current_thread().name}] {response_text}")
        print(f"[{threading.current_thread().name}] ---------------------------------")
        return response_text
    except Exception as e:
        print(f"[{threading.current_thread().name}] ❗ Error during Gemini analysis for {clip_filename}: {e}")
        return f"Category: Error\nSummary: Failed to analyze due to: {e}"

def categorize_and_move_clip(clip_path: str, gemini_output: str):
    """
    Parses Gemini's output to determine the category and moves the clip to the corresponding folder.
    """
    category = "General News" # Default category

    match = re.search(r"Category:\s*(.+?)(?:\n|$)", gemini_output, re.IGNORECASE)
    if match:
        extracted_category = match.group(1).strip().replace(" ", "_")
        if extracted_category.lower() == "weather":
            category = "Weather"
        elif extracted_category.lower() == "breaking_news":
            category = "Breaking_News"
        elif extracted_category.lower() == "general_news":
            category = "General_News"
        else:
            category = "Other_Events"

    destination_dir = os.path.join(CATEGORIZED_CLIPS_DIR, category)
    os.makedirs(destination_dir, exist_ok=True)

    try:
        clip_filename = os.path.basename(clip_path)
        destination_path = os.path.join(destination_dir, clip_filename)

        shutil.move(clip_path, destination_path)
        print(f"[{threading.current_thread().name}] Moved '{clip_filename}' to '{destination_dir}'")
    except FileNotFoundError:
        print(f"[{threading.current_thread().name}] Error: Clip '{clip_filename}' not found at '{clip_path}' for moving.")
    except Exception as e:
        print(f"[{threading.current_thread().name}] Error moving clip '{clip_filename}' to '{destination_dir}': {e}")


def clip_processing_worker(worker_id: int, whisper_model_instance):
    """
    Worker thread that pulls clip paths from the queue, processes them,
    and then ends its task for that specific clip.
    """
    thread_name = f"ClipProcessor-{worker_id}"
    print(f"[{thread_name}] started.")
    while not stop_event.is_set() or not clip_queue.empty():
        try:
            clip_path, clip_filename, timestamp_str, clip_counter = clip_queue.get(timeout=1)
            print(f"[{thread_name}] picked up clip: {clip_filename}")

            transcript_filename = f"transcript_{timestamp_str}_{clip_counter:03d}.txt"
            transcript_path = os.path.join(OUTPUT_TRANSCRIPTS_DIR, transcript_filename)

            gemini_raw_output = ""
            try:
                transcript_text = transcription(whisper_model_instance, clip_path, transcript_path, clip_filename)

                if transcript_text:
                    gemini_raw_output = gemini_analysis(transcript_text, clip_filename)
                else:
                    print(f"[{thread_name}] Skipping Gemini analysis for {clip_filename} due to empty transcript.")

            except Exception as e:
                print(f"[{thread_name}] ❗ Error during transcription/analysis of clip {clip_filename}: {e}")
            finally:
                if gemini_raw_output:
                    categorize_and_move_clip(clip_path, gemini_raw_output)
                else:
                    # Move to General_News if analysis failed or was skipped
                    if os.path.exists(clip_path):
                        default_dest = os.path.join(GENERAL_NEWS_CLIPS_DIR, os.path.basename(clip_path))
                        os.makedirs(GENERAL_NEWS_CLIPS_DIR, exist_ok=True) # Ensure it exists
                        try:
                            shutil.move(clip_path, default_dest)
                            print(f"[{thread_name}] Moved unprocessed clip '{os.path.basename(clip_path)}' to '{GENERAL_NEWS_CLIPS_DIR}'")
                        except Exception as e_move:
                            print(f"[{thread_name}] Error moving unprocessed clip '{os.path.basename(clip_path)}': {e_move}")

                clip_queue.task_done()
                print(f"[{thread_name}] finished processing clip: {clip_filename}")

        except queue.Empty:
            if stop_event.is_set():
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"[{thread_name}] An unexpected error occurred in worker: {e}")
            if stop_event.is_set():
                break

    print(f"[{thread_name}] stopped.")


def live_stream_cutter_thread(real_stream_url: str, clip_duration: int):
    """
    This thread continuously captures the live stream, cuts it into X-second
    clips, and puts the clip paths into a queue for worker threads to process.
    """
    print("[Cutter] Initiating stream cutting...")

    session = Streamlink()
    try:
        streams = session.streams(real_stream_url)
        if 'best' not in streams:
            print("[Cutter] No playable streams found.")
            stop_event.set()
            return
        stream = streams['best']
        fd = stream.open()
        print(f"[Cutter] Opening streamlink session for {real_stream_url}...")
    except Exception as e:
        print(f"[Cutter] Error opening streamlink session: {e}")
        stop_event.set()
        return

    clip_counter = 0
    buffer_data = b""
    start_time = time.time()

    print(f"[Cutter] Starting continuous recording of {clip_duration}-second clips...")

    try:
        while not stop_event.is_set():
            data = fd.read(8192)
            if not data:
                print("[Cutter] Stream ended or interrupted.")
                break

            buffer_data += data

            if (time.time() - start_time) >= clip_duration:
                clip_counter += 1
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                clip_filename = f"clip_{timestamp_str}_{clip_counter:03d}.ts"
                clip_path = os.path.join(OUTPUT_CLIPS_DIR, clip_filename)

                with open(clip_path, 'wb') as outfile:
                    outfile.write(buffer_data)
                print(f"[Cutter] ✔ Saved clip: {clip_path}")

                time.sleep(0.5)

                clip_queue.put((clip_path, clip_filename, timestamp_str, clip_counter))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cutter] Enqueued clip: {clip_filename} for processing.")

                buffer_data = b""
                start_time = time.time()
            else:
                # Add a small sleep to avoid busy-waiting if not enough data for a full clip yet
                time.sleep(0.01)

    except Exception as e:
        print(f"[Cutter] An unexpected error occurred during cutting: {e}")
    finally:
        if 'fd' in locals() and fd:
            fd.close()
            print("[Cutter] Streamlink session closed.")
        stop_event.set() # Signal other threads to stop if cutter fails or finishes
        print("[Cutter] stopped.")


def main():
    global WHISPER_MODEL

    print("Starting application...")
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(WEATHER_CLIPS_DIR, exist_ok=True)
    os.makedirs(BREAKING_NEWS_CLIPS_DIR, exist_ok=True)
    os.makedirs(GENERAL_NEWS_CLIPS_DIR, exist_ok=True)
    os.makedirs(OTHER_EVENTS_CLIPS_DIR, exist_ok=True)
    print(f"Created output directories: {OUTPUT_CLIPS_DIR}, {OUTPUT_TRANSCRIPTS_DIR}, and {CATEGORIZED_CLIPS_DIR} subfolders.")

    print("[Main] Loading Whisper 'small' model (once)...")
    try:
        WHISPER_MODEL = whisper.load_model("small")
        print("[Main] ✔ Whisper 'small' model loaded successfully.")
    except Exception as e:
        print(f"[Main] Fatal Error: Could not load Whisper model: {e}")
        return

    url = "https://www.wusa9.com/watch"
    try:
        real_stream_url = realstream(url, 15)
    except RuntimeError as e:
        print(f"[Main] Fatal Error: {e}")
        return

    cutter_thread = threading.Thread(
        target=live_stream_cutter_thread,
        args=(real_stream_url, CLIP_DURATION_SECONDS),
        name="StreamCutter"
    )
    cutter_thread.start()

    worker_threads = []
    for i in range(NUM_WORKER_THREADS):
        worker = threading.Thread(
            target=clip_processing_worker,
            args=(i + 1, WHISPER_MODEL),
            name=f"ClipProcessor-{i+1}"
        )
        worker_threads.append(worker)
        worker.start()

    print(f"[Main] Started {NUM_WORKER_THREADS} worker threads for clip processing.")

    # Start the timer for automatic shutdown
    if RUNTIME_MINUTES > 0:
        shutdown_delay_seconds = RUNTIME_MINUTES * 60
        shutdown_timer = threading.Timer(shutdown_delay_seconds, stop_event.set)
        shutdown_timer.start()
        print(f"[Main] Automatic shutdown scheduled in {RUNTIME_MINUTES} minutes.")

    try:
        while not stop_event.is_set():
            time.sleep(1) # Main thread sleeps and checks stop_event periodically

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Signaling threads to stop...")
        stop_event.set()

    finally:
        # If the timer was started, ensure it's cancelled if we shut down early
        if 'shutdown_timer' in locals() and shutdown_timer.is_alive():
            shutdown_timer.cancel()
            print("[Main] Shutdown timer cancelled.")

        print("[Main] Waiting for StreamCutter to finish...")
        cutter_thread.join()

        print("[Main] Waiting for pending clips to be processed by workers...")
        clip_queue.join() # This will wait until all items put into the queue are processed

        # A final signal to workers, in case they were waiting on queue.Empty and stop_event wasn't set yet
        stop_event.set()

        for worker in worker_threads:
            print(f"[Main] Waiting for {worker.name} to finish...")
            worker.join()

        print("[Main] All threads stopped. Application gracefully terminated.")


if __name__ == '__main__':
    main()