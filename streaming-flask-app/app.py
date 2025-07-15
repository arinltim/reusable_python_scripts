import os
import shutil
import threading
import queue
import time
import json
import logging
from datetime import datetime
import ffmpeg
import werkzeug.utils
import re

# New imports for audio analysis and hashing
import numpy as np
import soundfile as sf
import imagehash
from PIL import Image

from flask import Flask, render_template, jsonify, send_from_directory, abort, Response, request
import google.generativeai as genai
from streamlink import Streamlink
from waitress import serve

# Add this import to silence the noisy but harmless SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- CONFIGURATION ---
LIVE_BUFFER_SECONDS = 60 # Capture a 60-second buffer to find scenes within it
OUTPUT_CLIPS_DIR = "output_clips_live"
CATEGORIZED_CLIPS_DIR = "static/categorized_clips_live"
HASH_DIFFERENCE_THRESHOLD = 10
AUDIO_SPIKE_THRESHOLD = 0.15 # Threshold for detecting a "cheer" (adjust as needed)


# --- FLASK APP SETUP & GLOBALS ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')
stop_event = threading.Event()
new_clips_list = []
new_clips_lock = threading.Lock()
processing_thread_instance = None
processing_thread_lock = threading.Lock()
processed_hashes = set()
hash_lock = threading.Lock()


# ==============================================================================
# SECTION 1: CORE PROCESSING LOGIC
# ==============================================================================

def stream_processor_thread(stream_url: str, headers: dict):
    """
    This single thread uses the "Audio Watchdog" method for high-speed event detection.
    """
    logging.info(f"✅ [PROCESSOR]: Thread started for {stream_url}")

    clip_counter = 0
    while not stop_event.is_set():
        try:
            logging.info("Attempting to connect to stream...")
            session = Streamlink()
            session.set_option("http-headers", headers)
            session.set_option("hls-live-edge", 4)
            session.set_option("hls-timeout", 60)
            session.set_option("http-ssl-verify", False)

            streams = session.streams(stream_url)
            if 'best' not in streams:
                logging.error("❌ [PROCESSOR]: 'best' quality stream not found. Retrying..."); time.sleep(30); continue

            with streams['best'].open() as fd:
                logging.info("✅ [PROCESSOR]: Stream opened. Entering capture loop.")
                while not stop_event.is_set():
                    buffer_ts_path = None
                    try:
                        buffer_ts_path = os.path.join(OUTPUT_CLIPS_DIR, f"buffer_{clip_counter}.ts")
                        logging.info(f"Capturing {LIVE_BUFFER_SECONDS}s buffer #{clip_counter}...")
                        with open(buffer_ts_path, 'wb') as outfile:
                            start_time = time.time()
                            while time.time() - start_time < LIVE_BUFFER_SECONDS:
                                if stop_event.is_set(): break
                                data = fd.read(1024*1024) # Read in 1MB chunks
                                if not data: raise StopIteration
                                outfile.write(data)

                        if stop_event.is_set(): break

                        # --- [WATCHDOG] Analyze audio for exciting moments ---
                        logging.info(f"Analyzing audio for buffer #{clip_counter}...")
                        exciting_timestamps = find_audio_spikes(buffer_ts_path, threshold=AUDIO_SPIKE_THRESHOLD)

                        if not exciting_timestamps:
                            logging.info("No loud audio events found in buffer.")
                            continue

                        logging.info(f"Detected {len(exciting_timestamps)} potential highlight(s) in buffer!")

                        # --- Process each exciting moment found ---
                        for timestamp in exciting_timestamps:
                            highlight_path, thumbnail_path = None, None
                            try:
                                pre_roll, post_roll = 5, 20 # Create a 25-second highlight
                                # print(timestamp)
                                highlight_start = max(0, timestamp - pre_roll)
                                # print(highlight_start)
                                highlight_duration = pre_roll + post_roll
                                highlight_filename = f"highlight_{int(time.time())}_{clip_counter}.mp4"
                                highlight_path = os.path.join(OUTPUT_CLIPS_DIR, highlight_filename)

                                logging.info(f"Creating {highlight_duration}s highlight clip around {timestamp:.2f}s...")
                                # ffmpeg.input(buffer_ts_path, ss=highlight_start).output(highlight_path, t=highlight_duration, vcodec='libx264', acodec='aac', g=30).run(overwrite_output=True, quiet=True)
                                ffmpeg.input(buffer_ts_path).output(
                                    highlight_path,
                                    ss=highlight_start, # Moved here
                                    t=highlight_duration,
                                    vcodec='libx264',
                                    acodec='aac',
                                    g=30
                                ).run(overwrite_output=True, quiet=True)

                                # Send this confirmed highlight to Gemini for the final summary
                                gemini_output = gemini_video_analysis(highlight_path)
                                if not gemini_output: continue

                                analysis = json.loads(re.sub(r',\s*([}\]])', r'\1', gemini_output))
                                event = analysis.get("event", "Highlight")
                                summary = analysis.get("summary", "An exciting moment was detected.")

                                thumbnail_path = extract_thumbnail(highlight_path, seek_time=pre_roll)
                                if not thumbnail_path: continue

                                new_hash = generate_hash(thumbnail_path)
                                if new_hash is None: continue

                                is_duplicate = False
                                with hash_lock:
                                    if any(new_hash - old_hash < HASH_DIFFERENCE_THRESHOLD for old_hash in processed_hashes):
                                        is_duplicate = True
                                    else:
                                        processed_hashes.add(new_hash)

                                if is_duplicate:
                                    logging.info(f"DUPLICATE DETECTED: Discarding highlight for '{event}'.")
                                    if os.path.exists(highlight_path): os.remove(highlight_path)
                                    if os.path.exists(thumbnail_path): os.remove(thumbnail_path)
                                else:
                                    handle_highlight_clip(highlight_path, thumbnail_path, summary, event)

                            except Exception as e:
                                logging.error(f"Error processing highlight at timestamp {timestamp}: {e}")

                    except StopIteration:
                        logging.warning("❌ [PROCESSOR]: Stream data stopped. Attempting to reconnect."); break
                    except Exception as e:
                        logging.error(f"Error during processing loop for buffer {clip_counter}: {e}")
                    finally:
                        if os.path.exists(buffer_ts_path): os.remove(buffer_ts_path)
                        clip_counter += 1
        except Exception as e:
            logging.error(f"❌ [PROCESSOR]: A critical connection error occurred: {e}. Retrying in 30 seconds...")
            time.sleep(30)

    logging.info(f"✅ [PROCESSOR]: Stop event received. Thread is finishing.")

def find_audio_spikes(video_path, threshold=0.15, duration=1.0, min_gap=15):
    """
    Analyzes a video file's audio track to find timestamps of volume spikes.
    Returns a list of timestamps (in seconds) where spikes occur.
    """
    temp_audio_file = video_path + ".wav"
    spike_timestamps = []
    try:
        # Extract audio to a temporary WAV file
        ffmpeg.input(video_path).output(temp_audio_file, acodec='pcm_s16le', ar='16000', ac=1).run(overwrite_output=True, quiet=True)

        audio_data, samplerate = sf.read(temp_audio_file)

        # Handle empty or silent audio files gracefully
        if len(audio_data) == 0:
            return []

        # Normalize audio data to a -1.0 to 1.0 range
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 0:
            audio_data = audio_data / max_abs

        samples_per_window = int(samplerate * duration)

        # [NEW] Log the maximum volume found in the entire clip for easy tuning
        max_rms = np.sqrt(np.mean(audio_data**2))
        logging.info(f"Audio analysis for {os.path.basename(video_path)} - Max RMS volume: {max_rms:.4f}")

        last_spike_time = -min_gap

        for i in range(0, len(audio_data) - samples_per_window, samples_per_window):
            window = audio_data[i:i+samples_per_window]
            rms = np.sqrt(np.mean(window**2))

            current_time = i / samplerate

            if rms > threshold and (current_time - last_spike_time) > min_gap:
                timestamp = current_time + (duration / 2)
                spike_timestamps.append(timestamp)
                last_spike_time = timestamp
                logging.info(f"Loud audio spike DETECTED at ~{timestamp:.2f}s (RMS: {rms:.2f})")

    except Exception as e:
        logging.error(f"Could not analyze audio for {video_path}: {e}")
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

    return spike_timestamps


def gemini_video_analysis(clip_path: str) -> str | None:
    """Sends a confirmed highlight clip to Gemini for a descriptive summary."""
    try:
        system_instruction = "You are a world-class sports commentator AI..."
        model = genai.GenerativeModel("gemini-1.5-pro-latest", system_instruction=system_instruction)

        with open(clip_path, 'rb') as f: video_bytes = f.read()
        video_part = {"mime_type": "video/mp4", "data": video_bytes}

        text_prompt = """
            Analyze this short soccer highlight clip. The clip is already confirmed to be an exciting moment like a goal, save, or major foul. Your task is to provide the final classification and a concise, exciting summary.
            Your output MUST be a single, valid JSON object with these keys:
            - "event": Classify the event as "Goal", "Save", "Shot Miss", or "Foul".
            - "summary": A one-sentence, broadcast-style summary of the highlight.
            """

        response = model.generate_content([text_prompt, video_part], request_options={"timeout": 120})

        if response.text:
            json_start_index = response.text.find('{')
            if json_start_index != -1: return response.text[json_start_index:].strip().replace("```", "")
        return None
    except Exception as e:
        logging.error(f"❗ An exception occurred during Gemini video analysis: {e}")
        return None

def handle_highlight_clip(video_path, thumb_path, summary, category):
    safe_category_dir = os.path.join(CATEGORIZED_CLIPS_DIR, category.replace(" ", "_"))
    os.makedirs(safe_category_dir, exist_ok=True)
    video_filename, thumb_filename = os.path.basename(video_path), os.path.basename(thumb_path)
    try:
        shutil.move(video_path, os.path.join(safe_category_dir, video_filename))
        shutil.move(thumb_path, os.path.join(safe_category_dir, thumb_filename))
        video_web_path = f"{category.replace(' ', '_')}/{video_filename}"
        thumb_web_path = f"{category.replace(' ', '_')}/{thumb_filename}"
        clip_data = {"video_path": video_web_path, "thumb_path": thumb_web_path, "category": category, "summary": summary, "timestamp": datetime.now().strftime('%H:%M:%S')}
        with new_clips_lock:
            new_clips_list.append(clip_data)
        logging.info(f"✔ Published '{category}' event: {video_filename}")
    except Exception as e:
        logging.error(f"Error handling highlight clip for '{video_filename}': {e}")

def extract_thumbnail(mp4_path, seek_time=1.0):
    thumbnail_path = os.path.join(os.path.dirname(mp4_path), os.path.basename(mp4_path).replace(".mp4", ".jpg"))
    try:
        ffmpeg.input(mp4_path, ss=seek_time).output(thumbnail_path, vframes=1).run(overwrite_output=True, quiet=True)
        return thumbnail_path
    except ffmpeg.Error: return None

def generate_hash(image_path):
    try:
        with Image.open(image_path) as img: return imagehash.phash(img)
    except Exception: return None

# --- FLASK ROUTES & STARTUP ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/static/categorized_clips_live/<category>/<filename>')
def serve_clip(category, filename):
    try:
        directory = os.path.join(os.getcwd(), "static", "categorized_clips_live", werkzeug.utils.secure_filename(category))
        return send_from_directory(directory, werkzeug.utils.secure_filename(filename))
    except FileNotFoundError: abort(404)

@app.route('/get-updates')
def get_updates():
    global new_clips_list
    with new_clips_lock:
        clips_to_send = new_clips_list[:]
        new_clips_list.clear()
    return jsonify(clips_to_send)

@app.route('/start-processing', methods=['POST'])
def start_processing_route():
    global processing_thread_instance
    with processing_thread_lock:
        if processing_thread_instance and processing_thread_instance.is_alive():
            return jsonify({"status": "error", "message": "Processing already in progress."}), 409

        stop_event.clear()
        with new_clips_lock: new_clips_list.clear()
        with hash_lock: processed_hashes.clear()

        shutil.rmtree(OUTPUT_CLIPS_DIR, ignore_errors=True); os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
        shutil.rmtree(CATEGORIZED_CLIPS_DIR, ignore_errors=True); os.makedirs(CATEGORIZED_CLIPS_DIR, exist_ok=True)

        data = request.get_json()
        url = data.get('stream_url')
        if not url: return jsonify({"status": "error", "message": "stream_url not provided."}), 400

        headers = {"User-Agent": data.get("user_agent", "Mozilla/5.0"), "Referer": data.get("referer", url)}

        processing_thread_instance = threading.Thread(target=stream_processor_thread, args=(url, headers), name="StreamProcessor")
        processing_thread_instance.start()

        return jsonify({"status": "success", "message": "Live stream processing started."})

@app.route('/stop-processing', methods=['POST'])
def stop_processing_route():
    global processing_thread_instance
    with processing_thread_lock:
        stop_event.set()
        if processing_thread_instance:
            processing_thread_instance.join(timeout=10)
        processing_thread_instance = None
        logging.info("Processing stopped by user command.")
        return jsonify({"status": "success", "message": "Processing stopped."})

if __name__ == '__main__':
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.critical("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
    else:
        try:
            genai.configure(api_key=api_key)
            logging.info("✔ Gemini API client configured successfully.")
            serve(app, host='localhost', port=5001, threads=8)
        except Exception as e:
            logging.critical(f"FATAL ERROR: Failed to configure Gemini API client: {e}")
