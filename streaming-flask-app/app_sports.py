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

from flask import Flask, render_template, jsonify, send_from_directory, abort, Response
import google.generativeai as genai

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# --- CONFIGURATION ---
SOURCE_VIDEO_FILE = os.path.join("static", "USA-vs-Mexico.mp4")
OUTPUT_CLIPS_DIR = "output_clips_sports"
CATEGORIZED_CLIPS_DIR = os.path.join("static", "categorized_clips_sports")
NUM_WORKER_THREADS = 4
HASH_DIFFERENCE_THRESHOLD = 5 # Stricter deduplication

# --- FLASK APP SETUP & GLOBALS ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')
stop_event = threading.Event()
clip_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)
sse_queue = queue.Queue()
captured_clips_info = []
captured_clips_lock = threading.Lock()
highlight_counter = 0
counter_lock = threading.Lock()
processed_hashes = set()
hash_lock = threading.Lock()

# ==============================================================================
# SECTION 1: VIDEO & AI PROCESSING
# ==============================================================================

def process_video_file_thread(source_path: str):
    if not os.path.exists(source_path):
        logging.error(f"FATAL: Source video file not found at '{source_path}'"); return
    logging.info("Starting scene detection process...")
    try:
        video = open_video(source_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        logging.info(f"Scene detection complete. Found {len(scene_list)} scenes.")
        for i, scene in enumerate(scene_list):
            if stop_event.is_set(): break
            start_time, end_time = scene
            start_sec, duration_sec = start_time.get_seconds(), end_time.get_seconds() - start_time.get_seconds()
            if duration_sec < 2.5: continue
            scene_filename = f"scene_{i:04d}.mp4"
            scene_path = os.path.join(OUTPUT_CLIPS_DIR, scene_filename)
            logging.info(f"Cutting scene {i}: from {start_time} to {end_time} ({duration_sec:.2f}s)")
            ffmpeg.input(source_path, ss=start_sec, t=duration_sec).output(scene_path, acodec='copy', vcodec='copy').run(overwrite_output=True, quiet=True)
            clip_queue.put((scene_path, start_sec))
    except Exception as e:
        logging.error(f"An error occurred during scene detection or cutting: {e}")
    finally:
        logging.info("Finished processing all scenes.")
        stop_event.set()

def gemini_video_analysis(clip_path: str) -> str | None:
    """
    [NEW] This version uses a system instruction and asks for player details.
    """
    try:
        system_instruction = "You are a helpful assistant designed to output JSON. Your entire response must always be a single, valid JSON object. Do not under any circumstances output plain text or markdown."
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
        with open(clip_path, 'rb') as f: video_bytes = f.read()
        video_part = {"mime_type": "video/mp4", "data": video_bytes}
        text_prompt = f"""
            Analyze this soccer video clip from a game. Classify the most significant event and provide details.
            
            The JSON object you output must contain the following keys:
            - "event": "Goal", "Shot on Target", "Shot Miss", "Foul", "Penalty Kick", "Save", or "No Significant Event". If no clear action is visible, you MUST use "No Significant Event".
            - "timestamp_in_clip": A number in seconds. Required if an event is found.
            - "summary": A one-sentence description of the event.
            - "player_jersey_color": The jersey color of the key player. Use "Unknown" if not visible.
            - "player_jersey_number": The jersey number of the key player. Use "Unknown" if not visible.
            """
        prompt_parts = [text_prompt, video_part]
        response = model.generate_content(prompt_parts, request_options={"timeout": 120})

        if response.text:
            return response.text.strip().replace("```json", "").replace("```", "")
        else:
            logging.warning(f"Gemini API returned an empty response for {os.path.basename(clip_path)}.")
            return None
    except Exception as e:
        logging.error(f"❗ An exception occurred during Gemini video analysis for {os.path.basename(clip_path)}: {e}")
        return None

def parse_timestamp(value) -> float | None:
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try:
            parts = value.split(':');
            if len(parts) == 2: return float(int(parts[0]) * 60 + int(parts[1]))
            return float(value)
        except (ValueError, TypeError): return None
    return None

def clip_processing_worker():
    global highlight_counter
    while not stop_event.is_set() or not clip_queue.empty():
        try:
            chunk_path, chunk_start_time = clip_queue.get(timeout=1)
        except queue.Empty:
            continue

        time.sleep(0.5) # Prevent API rate-limiting

        final_highlight_path, thumbnail_path = None, None
        try:
            gemini_output = gemini_video_analysis(chunk_path)
            if not gemini_output:
                logging.warning(f"Skipping chunk {os.path.basename(chunk_path)} due to analysis failure.")
                continue

            cleaned_json_string = re.sub(r',\s*([}\]])', r'\1', gemini_output)
            analysis = json.loads(cleaned_json_string)
            event = analysis.get("event")

            interesting_events = ["Goal", "Shot on Target", "Shot Miss", "Foul", "Penalty Kick", "Save"]

            if event in interesting_events:
                timestamp_in_clip = parse_timestamp(analysis.get("timestamp_in_clip"))
                if timestamp_in_clip is None: continue

                with counter_lock:
                    highlight_counter += 1; current_highlight_id = highlight_counter

                absolute_action_time = chunk_start_time + timestamp_in_clip
                pre_action_seconds = 5; post_action_seconds = 5
                highlight_start_time = max(0, absolute_action_time - pre_action_seconds)
                new_highlight_duration = pre_action_seconds + post_action_seconds

                highlight_filename = f"highlight_{event.replace(' ', '_')}_{current_highlight_id:03d}.mp4"
                destination_dir = os.path.join(CATEGORIZED_CLIPS_DIR, event.replace(" ", "_"))
                os.makedirs(destination_dir, exist_ok=True)
                final_highlight_path = os.path.join(destination_dir, highlight_filename)

                ffmpeg.input(SOURCE_VIDEO_FILE, ss=highlight_start_time, t=new_highlight_duration).output(final_highlight_path).run(overwrite_output=True, quiet=True)
                thumbnail_path = extract_thumbnail(final_highlight_path, seek_time=pre_action_seconds)

                if not thumbnail_path: raise ValueError("Could not extract thumbnail for hashing.")

                new_hash = generate_hash(thumbnail_path)
                if new_hash is None: raise ValueError("Could not generate hash for thumbnail.")

                is_duplicate = False
                with hash_lock:
                    for existing_hash in processed_hashes:
                        if new_hash - existing_hash < HASH_DIFFERENCE_THRESHOLD:
                            is_duplicate = True; break
                    if not is_duplicate:
                        processed_hashes.add(new_hash)

                if is_duplicate:
                    logging.info(f"DUPLICATE DETECTED: Discarding clip for '{event}' (ID: {current_highlight_id}).")
                    if os.path.exists(final_highlight_path): os.remove(final_highlight_path)
                    if os.path.exists(thumbnail_path): os.remove(thumbnail_path)
                    continue

                summary = analysis.get("summary", "An interesting event was detected.")
                jersey_color = analysis.get("player_jersey_color")
                jersey_number = analysis.get("player_jersey_number")

                if jersey_color and jersey_color != "Unknown":
                    player_id = f"Player #{jersey_number}" if jersey_number and jersey_number != "Unknown" else f"A player in {jersey_color}"
                    summary = f"({player_id}) {summary}"

                handle_highlight_clip(final_highlight_path, thumbnail_path, summary, event)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from Gemini API for {os.path.basename(chunk_path)}. Response was: '{gemini_output}'")
        except Exception as e:
            logging.error(f"Error processing scene chunk {os.path.basename(chunk_path)}: {e}")
        finally:
            if os.path.exists(chunk_path): os.remove(chunk_path)
            clip_queue.task_done()

# Other helper functions...
def handle_highlight_clip(video_path, thumb_path, summary, category):
    safe_category_dir = category.replace(" ", "_")
    video_filename, thumb_filename = os.path.basename(video_path), os.path.basename(thumb_path)
    final_thumb_path = os.path.join(CATEGORIZED_CLIPS_DIR, safe_category_dir, thumb_filename)
    try:
        shutil.move(thumb_path, final_thumb_path)
        video_web_path = f"{safe_category_dir}/{video_filename}"
        thumb_web_path = f"{safe_category_dir}/{thumb_filename}"
        clip_data = {"video_path": video_web_path, "thumb_path": thumb_web_path, "category": category, "summary": summary, "timestamp": datetime.now().strftime('%H:%M:%S')}
        sse_queue.put(clip_data)
        with captured_clips_lock: captured_clips_info.append(clip_data)
        logging.info(f"✔ Published '{category}' event: {video_filename}")
    except Exception as e:
        logging.error(f"Error handling highlight clip for '{video_filename}': {e}")

def extract_thumbnail(mp4_path, seek_time=1.0):
    thumbnail_path = os.path.join(OUTPUT_CLIPS_DIR, os.path.basename(mp4_path).replace(".mp4", ".jpg"))
    try:
        ffmpeg.input(mp4_path, ss=seek_time).output(thumbnail_path, vframes=1).run(overwrite_output=True, quiet=True)
        return thumbnail_path
    except ffmpeg.Error:
        logging.error(f"Could not extract thumbnail for {mp4_path}")
        return None

def generate_hash(image_path):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        logging.error(f"Could not generate hash for image {image_path}: {e}")
        return None

# --- FLASK ROUTES & STARTUP ---
@app.route('/')
def index(): return render_template('index_sports.html')

@app.route('/initial-clips')
def initial_clips():
    with captured_clips_lock: return jsonify(captured_clips_info)

@app.route('/categorized_clips_sports/<category>/<filename>')
def serve_clip(category, filename):
    safe_category, safe_filename = werkzeug.utils.secure_filename(category), werkzeug.utils.secure_filename(filename)
    if safe_category != category or safe_filename != filename: abort(404)
    try:
        directory = os.path.join(os.getcwd(), "static", "categorized_clips_sports", safe_category)
        return send_from_directory(directory, safe_filename)
    except FileNotFoundError: abort(404)

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            try: data = sse_queue.get_nowait(); yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty: pass
            yield ": keep-alive\n\n"; time.sleep(1)
    return Response(event_stream(), mimetype='text/event-stream')

def start_background_tasks():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.critical("FATAL ERROR: GEMINI_API_KEY environment variable not set. Application cannot start processing.")
        return
    try:
        genai.configure(api_key=api_key)
        logging.info("✔ Gemini API client configured successfully.")
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to configure Gemini API client: {e}")
        return

    shutil.rmtree(OUTPUT_CLIPS_DIR, ignore_errors=True)
    shutil.rmtree(CATEGORIZED_CLIPS_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(CATEGORIZED_CLIPS_DIR, exist_ok=True)
    with hash_lock:
        processed_hashes.clear()

    producer_thread = threading.Thread(target=process_video_file_thread, args=(SOURCE_VIDEO_FILE,), name="VideoChunker", daemon=True)
    producer_thread.start()
    for i in range(NUM_WORKER_THREADS):
        worker = threading.Thread(target=clip_processing_worker, name=f"ClipProcessor-{i+1}", daemon=True)
        worker.start()
    logging.info(f"Started {NUM_WORKER_THREADS} worker threads for soccer analysis.")

if __name__ == '__main__':
    init_thread = threading.Thread(target=start_background_tasks, daemon=True)
    init_thread.start()
    app.run(host='0.0.0.0', port=5001, threaded=True)