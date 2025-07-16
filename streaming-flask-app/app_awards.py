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

import imagehash
from PIL import Image

from flask import Flask, render_template, jsonify, send_from_directory, abort, Response, request
import google.generativeai as genai
from streamlink import Streamlink
from waitress import serve
import whisper

# Add this import to silence the noisy but harmless SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- CONFIGURATION ---
CLIP_DURATION_SECONDS = 30
OUTPUT_CLIPS_DIR = "output_clips_oscars_live"
CATEGORIZED_CLIPS_DIR = "static/categorized_clips_oscars_live"
NUM_WORKER_THREADS = 2 # Transcription is CPU intensive
HASH_DIFFERENCE_THRESHOLD = 10

# --- FLASK APP SETUP & GLOBALS ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')
stop_event = threading.Event()
clip_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)
new_clips_list = []
new_clips_lock = threading.Lock()
cutter_thread_instance = None
cutter_thread_lock = threading.Lock()
worker_threads = []
processed_winners = set()
winners_lock = threading.Lock()
WHISPER_MODEL = None


# ==============================================================================
# SECTION 1: CORE PROCESSING LOGIC
# ==============================================================================

def live_stream_cutter_thread(stream_url: str, headers: dict):
    """
    [CORRECTED] This version correctly sends a tuple (path, start_time) to
    the processing queue, fixing the "too many values to unpack" error.
    """
    logging.info(f"✅ [CUTTER_THREAD]: Thread started for {stream_url}")
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
                logging.error("❌ [CUTTER_THREAD]: 'best' quality stream not found. Retrying in 30s..."); time.sleep(30); continue

            with streams['best'].open() as fd:
                logging.info("✅ [CUTTER_THREAD]: Stream opened. Entering capture loop.")
                while not stop_event.is_set():
                    clip_path = os.path.join(OUTPUT_CLIPS_DIR, f"live_chunk_{clip_counter}.ts")
                    try:
                        with open(clip_path, 'wb') as outfile:
                            start_time = time.time()
                            while time.time() - start_time < CLIP_DURATION_SECONDS:
                                if stop_event.is_set(): break
                                data = fd.read(8192)
                                if not data: raise StopIteration
                                outfile.write(data)

                        if stop_event.is_set():
                            if os.path.exists(clip_path): os.remove(clip_path)
                            break

                        # [THE FIX] Calculate the chunk's start time and put both
                        # the path and the start time on the queue as a tuple.
                        chunk_start_time = clip_counter * CLIP_DURATION_SECONDS
                        clip_queue.put((clip_path, chunk_start_time))

                        logging.info(f"✅ [CUTTER_THREAD]: Enqueued chunk {clip_counter}"); clip_counter += 1
                    except StopIteration:
                        logging.warning("❌ [CUTTER_THREAD]: Stream data stopped. Attempting to reconnect."); break
        except Exception as e:
            logging.error(f"❌ [CUTTER_THREAD]: A critical connection error occurred: {e}. Retrying in 30s...")
            time.sleep(30)

    logging.info(f"✅ [CUTTER_THREAD]: Thread is finishing.")


def transcription(model, clip_path: str) -> str:
    """Uses Whisper to transcribe the audio from a video clip."""
    try:
        logging.info(f"→ Transcribing audio from: {os.path.basename(clip_path)}...")
        probe = ffmpeg.probe(clip_path)
        if not any(stream['codec_type'] == 'audio' for stream in probe.get('streams', [])):
            logging.warning(f"No audio stream found in {os.path.basename(clip_path)}. Skipping.")
            return ""

        result = model.transcribe(clip_path, fp16=False, language="en")
        text = result["text"].strip()
        if text: logging.info(f"✔ Transcription found for {os.path.basename(clip_path)}")
        return text
    except Exception as e:
        logging.warning(f"Could not transcribe {os.path.basename(clip_path)} (likely silent or corrupt). Error: {e}")
        return ""

def gemini_text_analysis(transcript: str) -> str | None:
    """Analyzes a transcript to find an Oscar winner announcement."""
    try:
        system_instruction = "You are an expert entertainment journalist. Your task is to find winner announcements in a transcript from an awards show. Your entire response must be a single, valid JSON object."
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
        generation_config = genai.GenerationConfig(temperature=0.1)

        text_prompt = f"""
            Analyze the following transcript from an awards show.
            Identify if a winner is being announced for a specific category. Look for phrases like "And the Oscar goes to...", "the winner is...", etc.

            If an announcement is found, your output MUST be a single JSON object with these keys:
            - "is_announcement": true
            - "award_category": The name of the award (e.g., "Best Actor", "Best Picture").
            - "winner_name": The name of the person or film that won.
            - "timestamp_in_clip": The time in seconds within this transcript chunk where the winner's name is first mentioned.

            If no winner announcement is found, you MUST return a JSON object with "is_announcement" set to false.

            Transcript:
            ---
            {transcript}
            ---
            """

        response = model.generate_content(text_prompt, generation_config=generation_config, request_options={"timeout": 60})

        if response.text:
            return response.text.strip().replace("```json", "").replace("```", "")
        return None
    except Exception as e:
        logging.error(f"❗ An exception occurred during Gemini text analysis: {e}")
        return None

def clip_processing_worker():
    """
    Worker thread that uses transcription and keyword spotting to find moments.
    """
    keywords = [
        "the winner is", "and the oscar goes to", "accepting the award for",
        "best actor", "best actress", "best picture", "best director"
    ]

    while not stop_event.is_set() or not clip_queue.empty():
        try:
            chunk_path, chunk_start_time = clip_queue.get(timeout=1)
        except queue.Empty:
            continue

        mp4_path, highlight_path, thumbnail_path = None, None, None
        try:
            mp4_path = convert_ts_to_mp4(chunk_path)
            if not mp4_path: continue

            transcript = transcription(WHISPER_MODEL, mp4_path)
            if not transcript or not any(keyword in transcript.lower() for keyword in keywords):
                continue

            logging.info(f"Keywords found in {os.path.basename(mp4_path)}. Sending to Gemini for confirmation...")
            gemini_output = gemini_text_analysis(transcript)
            if not gemini_output: continue

            analysis = json.loads(re.sub(r',\s*([}\]])', r'\1', gemini_output))

            if analysis.get("is_announcement"):
                winner = analysis.get("winner_name")
                category = analysis.get("award_category")
                timestamp_in_clip = parse_timestamp(analysis.get("timestamp_in_clip"))

                if not winner or not category or timestamp_in_clip is None: continue

                with winners_lock:
                    winner_key = f"{winner}_{category}".lower()
                    if winner_key in processed_winners:
                        logging.info(f"Duplicate winner found: {winner} for {category}. Skipping.")
                        continue
                    processed_winners.add(winner_key)

                pre_roll, post_roll = 5, 25
                highlight_start = max(0, timestamp_in_clip - pre_roll)
                highlight_duration = pre_roll + post_roll

                highlight_filename = f"highlight_{werkzeug.utils.secure_filename(winner)}_{int(time.time())}.mp4"

                # [FIX] Create the highlight in the temporary OUTPUT_CLIPS_DIR first
                highlight_path = os.path.join(OUTPUT_CLIPS_DIR, highlight_filename)

                logging.info(f"WINNER CONFIRMED: '{winner}' for '{category}'. Creating highlight clip...")
                ffmpeg.input(mp4_path, ss=highlight_start, t=highlight_duration).output(
                    highlight_path, vcodec='libx264', acodec='aac', g=30
                ).run(overwrite_output=True, quiet=True)

                thumbnail_path = extract_thumbnail(highlight_path, seek_time=pre_roll)
                if thumbnail_path:
                    # This function will now correctly move the files
                    handle_highlight_clip(highlight_path, thumbnail_path, category, winner)
                    # Set paths to None so they are not deleted by the finally block
                    highlight_path, thumbnail_path = None, None

        except Exception as e:
            logging.error(f"Error processing chunk {os.path.basename(chunk_path)}: {e}")
        finally:
            # This block now correctly cleans up all temporary files
            if mp4_path and os.path.exists(mp4_path): os.remove(mp4_path)
            if highlight_path and os.path.exists(highlight_path): os.remove(highlight_path)
            if thumbnail_path and os.path.exists(thumbnail_path): os.remove(thumbnail_path)
            if os.path.exists(chunk_path): os.remove(chunk_path)
            clip_queue.task_done()

def convert_ts_to_mp4(ts_path: str) -> str | None:
    mp4_path = ts_path.replace(".ts", ".mp4")
    try:
        ffmpeg.input(ts_path).output(mp4_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', strict='experimental').run(overwrite_output=True, quiet=True)
        return mp4_path
    except ffmpeg.Error: return None

def handle_highlight_clip(video_path, thumb_path, category, winner):
    """
    [CORRECTED] This function now correctly moves the highlight files to the
    final destination folder before notifying the UI.
    """
    safe_category_dir = os.path.join(CATEGORIZED_CLIPS_DIR, category.replace(" & ", "_and_").replace(" ", "_"))
    os.makedirs(safe_category_dir, exist_ok=True)

    video_filename = os.path.basename(video_path)
    thumb_filename = os.path.basename(thumb_path)

    final_video_path = os.path.join(safe_category_dir, video_filename)
    final_thumb_path = os.path.join(safe_category_dir, thumb_filename)

    try:
        # [THE FIX] Move the files from the temporary dir to the final static dir
        shutil.move(video_path, final_video_path)
        shutil.move(thumb_path, final_thumb_path)

        # The web paths for the UI are now relative to the CATEGORIZED_CLIPS_DIR
        video_web_path = f"{category.replace(' & ', '_and_').replace(' ', '_')}/{video_filename}"
        thumb_web_path = f"{category.replace(' & ', '_and_').replace(' ', '_')}/{thumb_filename}"

        clip_data = {
            "video_path": video_web_path,
            "thumb_path": thumb_web_path,
            "category": category,
            "winner": winner,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
        with new_clips_lock:
            new_clips_list.append(clip_data)
        logging.info(f"✔ Published highlight for winner: {winner}")

    except Exception as e:
        logging.error(f"Error handling highlight clip for '{video_filename}': {e}")

def extract_thumbnail(mp4_path, seek_time=1.0):
    thumbnail_path = mp4_path.replace(".mp4", ".jpg")
    try:
        ffmpeg.input(mp4_path, ss=seek_time).output(thumbnail_path, vframes=1).run(overwrite_output=True, quiet=True)
        return thumbnail_path
    except ffmpeg.Error: return None

def parse_timestamp(value) -> float | None:
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try:
            parts = value.split(':');
            if len(parts) == 2: return float(int(parts[0]) * 60 + int(parts[1]))
            return float(value)
        except (ValueError, TypeError): return None
    return None

# --- FLASK ROUTES & STARTUP ---
@app.route('/home')
def index(): return render_template('index_awards.html')

@app.route('/static/categorized_clips_oscars_live/<filename>')
def serve_clip(filename):
    return send_from_directory(CATEGORIZED_CLIPS_DIR, werkzeug.utils.secure_filename(filename))

@app.route('/get-updates')
def get_updates():
    global new_clips_list
    with new_clips_lock:
        clips_to_send = new_clips_list[:]
        new_clips_list.clear()
    return jsonify(clips_to_send)

@app.route('/start-processing', methods=['POST'])
def start_processing_route():
    global cutter_thread_instance, worker_threads
    with cutter_thread_lock:
        if cutter_thread_instance and cutter_thread_instance.is_alive():
            return jsonify({"status": "error", "message": "Processing already in progress."}), 409

        stop_event.clear()
        with new_clips_lock: new_clips_list.clear()
        with winners_lock: processed_winners.clear()

        shutil.rmtree(OUTPUT_CLIPS_DIR, ignore_errors=True); os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
        shutil.rmtree(CATEGORIZED_CLIPS_DIR, ignore_errors=True); os.makedirs(CATEGORIZED_CLIPS_DIR, exist_ok=True)

        data = request.get_json()
        url = data.get('stream_url')
        if not url: return jsonify({"status": "error", "message": "stream_url not provided."}), 400

        headers = {"User-Agent": data.get("user_agent", "Mozilla/5.0"), "Referer": data.get("referer", url)}

        cutter_thread_instance = threading.Thread(target=live_stream_cutter_thread, args=(url, headers), name="StreamCutter")
        cutter_thread_instance.start()

        worker_threads = []
        for i in range(NUM_WORKER_THREADS):
            worker = threading.Thread(target=clip_processing_worker, name=f"ClipProcessor-{i+1}")
            worker.start()
            worker_threads.append(worker)

        return jsonify({"status": "success", "message": "Live stream processing started."})

@app.route('/stop-processing', methods=['POST'])
def stop_processing_route():
    global cutter_thread_instance, worker_threads
    with cutter_thread_lock:
        stop_event.set()
        if cutter_thread_instance:
            cutter_thread_instance.join(timeout=10)

        for worker in worker_threads:
            worker.join(timeout=5)

        cutter_thread_instance = None
        worker_threads = []
        logging.info("Processing stopped by user command.")
        return jsonify({"status": "success", "message": "Processing stopped."})

def start_app_threads():
    """Loads models and prepares the app to run."""
    global WHISPER_MODEL
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.critical("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
        return False
    try:
        genai.configure(api_key=api_key)
        logging.info("✔ Gemini API client configured successfully.")
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to configure Gemini API client: {e}")
        return False

    logging.info("Loading Whisper 'base' model (this may take a moment)...")
    try:
        WHISPER_MODEL = whisper.load_model("base")
        logging.info("✔ Whisper model loaded successfully.")
    except Exception as e:
        logging.critical(f"FATAL: Could not load Whisper model: {e}")
        return False
    return True

if __name__ == '__main__':
    if start_app_threads():
        serve(app, host='localhost', port=5000, threads=16)
    else:
        logging.error("Application failed to initialize. Exiting.")
