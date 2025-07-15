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

from flask import Flask, render_template, jsonify, send_from_directory, abort, Response
import google.generativeai as genai

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# --- CONFIGURATION ---
SOURCE_VIDEO_FILE = os.path.join("static", "basketball_highlights.mp4")
# CHUNK_DURATION_SECONDS is no longer needed as we use scenes
OUTPUT_CLIPS_DIR = "output_clips_nba"
CATEGORIZED_CLIPS_DIR = os.path.join("static", "categorized_clips_nba")
SCORES_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Scores")
BLOCKS_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Blocks")
SHOT_MISSES_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Shot_Misses")
TURNOVERS_CLIPS_DIR = os.path.join(CATEGORIZED_CLIPS_DIR, "Turnovers")
NUM_WORKER_THREADS = 2

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

# ==============================================================================
# SECTION 1: SCENE-AWARE VIDEO PROCESSING
# ==============================================================================

def process_video_file_thread(source_path: str):
    """
    [CHANGED] This thread now uses PySceneDetect to create logical, scene-based
    chunks instead of fixed-duration ones.
    """
    if not os.path.exists(source_path):
        logging.error(f"FATAL: Source video file not found at '{source_path}'"); return

    logging.info("Starting scene detection process...")
    try:
        video = open_video(source_path)
        scene_manager = SceneManager()
        # Using ContentDetector to find fast cuts between camera angles.
        # The threshold determines sensitivity; a lower value means more scenes.
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        logging.info(f"Scene detection complete. Found {len(scene_list)} scenes.")

        for i, scene in enumerate(scene_list):
            if stop_event.is_set():
                logging.info("Stop event detected, halting scene processing.")
                break

            start_time, end_time = scene

            # Get timestamps in seconds
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()
            duration_sec = end_sec - start_sec

            # We can filter out very short scenes (e.g., flashes or quick cuts)
            if duration_sec < 2.0:
                continue

            scene_filename = f"scene_{i:04d}.mp4"
            scene_path = os.path.join(OUTPUT_CLIPS_DIR, scene_filename)

            logging.info(f"Cutting scene {i}: from {start_time} to {end_time} ({duration_sec:.2f}s)")

            # Use ffmpeg to cut the precise scene
            ffmpeg.input(source_path, ss=start_sec, t=duration_sec).output(
                scene_path, acodec='copy', vcodec='copy'
            ).run(overwrite_output=True, quiet=True)

            # Put the scene path and its start time onto the queue for analysis
            clip_queue.put((scene_path, start_sec))

    except Exception as e:
        logging.error(f"An error occurred during scene detection or cutting: {e}")
    finally:
        logging.info("Finished processing all scenes. Signaling workers to stop.")
        stop_event.set()


def gemini_video_analysis(clip_path: str) -> str:
    """Analyzes a video chunk using the new, more detailed prompt."""
    genai.configure(api_key="AIzaSyA6uP61fBK7dMirSEmAGwJpuGrKqsOIpX4")
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        with open(clip_path, 'rb') as f: video_bytes = f.read()
        video_part = {"mime_type": "video/mp4", "data": video_bytes}
        prompt = [
            f"""
            You are an expert NBA analyst AI. Your task is to meticulously analyze this basketball video clip, which represents a single continuous scene. Provide a detailed, accurate breakdown of the key event.
            Your output MUST be a single, valid JSON object.

            - Classify the main "event" as one of: "Score", "Block", "Shot Miss", "Turnover", or "No Event".
            - If an event occurs, you MUST provide a "timestamp_in_clip" (a number in seconds).
            - For all events, provide a descriptive one-sentence "summary".
            - For a "Score", you MUST also include a "shot_type" key ("Jump Shot", "Layup", "Dunk", "Free Throw").
            
            Base your conclusion on whether the ball goes through the hoop. A shot that touches the rim and goes in is still a "Score".

            EXAMPLE (Score):
            {{ "event": "Score", "timestamp_in_clip": 7.8, "summary": "Player #30 sinks a mid-range jump shot over the defender.", "shot_type": "Jump Shot" }}
            """,
            video_part
        ]
        response = model.generate_content(prompt, request_options={"timeout": 120})
        return response.text.strip().replace("```json", "").replace("```", "")
    except Exception as e:
        logging.error(f"❗ Error during Gemini video analysis: {e}")
        return json.dumps({"event": "Error"})


def parse_timestamp(value) -> float | None:
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try:
            parts = value.split(':')
            if len(parts) == 2:
                minutes = int(parts[0]); seconds = int(parts[1])
                return float(minutes * 60 + seconds)
            return float(value)
        except (ValueError, TypeError): return None
    return None

def clip_processing_worker():
    """Processes scene-based chunks and creates final, context-aware highlights."""
    global highlight_counter
    while not stop_event.is_set() or not clip_queue.empty():
        try:
            chunk_path, chunk_start_time = clip_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            gemini_output = gemini_video_analysis(chunk_path)
            cleaned_json_string = re.sub(r',\s*([}\]])', r'\1', gemini_output)
            analysis = json.loads(cleaned_json_string)
            event = analysis.get("event")
            interesting_events = ["Score", "Block", "Shot Miss", "Turnover"]

            if event in interesting_events:
                timestamp_in_clip = parse_timestamp(analysis.get("timestamp_in_clip"))
                summary = analysis.get("summary", "An interesting event was detected.")
                if timestamp_in_clip is None:
                    logging.warning(f"Event '{event}' detected but timestamp invalid. Skipping re-cut.")
                    continue

                with counter_lock:
                    highlight_counter += 1
                    current_highlight_id = highlight_counter

                absolute_action_time = chunk_start_time + timestamp_in_clip
                pre_action_seconds = 4; post_action_seconds = 6
                new_highlight_duration = pre_action_seconds + post_action_seconds
                highlight_start_time = max(0, absolute_action_time - pre_action_seconds)

                logging.info(f"EVENT DETECTED: '{event}'! Creating {new_highlight_duration}s highlight clip (ID: {current_highlight_id})...")

                highlight_filename = f"highlight_{event.replace(' ', '_')}_{current_highlight_id:03d}.mp4"
                safe_category_dir = event.replace(" ", "_")
                destination_dir = os.path.join(CATEGORIZED_CLIPS_DIR, safe_category_dir)
                os.makedirs(destination_dir, exist_ok=True)
                final_highlight_path = os.path.join(destination_dir, highlight_filename)

                ffmpeg.input(SOURCE_VIDEO_FILE, ss=highlight_start_time, t=new_highlight_duration).output(
                    final_highlight_path
                ).run(overwrite_output=True, quiet=True)

                thumbnail_path = extract_thumbnail(final_highlight_path, seek_time=pre_action_seconds)
                if thumbnail_path:
                    handle_highlight_clip(final_highlight_path, thumbnail_path, summary, event)

        except Exception as e:
            logging.error(f"Error processing scene chunk {os.path.basename(chunk_path)}: {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            clip_queue.task_done()

def handle_highlight_clip(video_path: str, thumb_path: str, summary: str, category: str):
    """Saves the final thumbnail and sends the event to the frontend."""
    safe_category_dir = category.replace(" ", "_")
    video_filename = os.path.basename(video_path)
    thumb_filename = os.path.basename(thumb_path)
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


def extract_thumbnail(mp4_path: str, seek_time: float = 1.0) -> str | None:
    thumbnail_path = os.path.join(OUTPUT_CLIPS_DIR, os.path.basename(mp4_path).replace(".mp4", ".jpg"))
    try:
        ffmpeg.input(mp4_path, ss=seek_time).output(thumbnail_path, vframes=1).run(overwrite_output=True, quiet=True)
        return thumbnail_path
    except ffmpeg.Error:
        logging.error(f"Could not extract thumbnail for {mp4_path}")
        return None

# --- FLASK ROUTES & STARTUP ---
@app.route('/')
def index(): return render_template('index_nba.html')
@app.route('/initial-clips')
def initial_clips():
    with captured_clips_lock: return jsonify(captured_clips_info)
@app.route('/categorized_clips_nba/<category>/<filename>')
def serve_clip(category, filename):
    safe_category = werkzeug.utils.secure_filename(category); safe_filename = werkzeug.utils.secure_filename(filename)
    if safe_category != category or safe_filename != filename: abort(404)
    try:
        directory = os.path.join(os.getcwd(), "static", "categorized_clips_nba", safe_category); return send_from_directory(directory, safe_filename)
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
    genai.configure(api_key="AIzaSyDUngYlgwM2HVLl4XQW1Px6-jRxDLwnEFU")
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(SCORES_CLIPS_DIR, exist_ok=True); os.makedirs(BLOCKS_CLIPS_DIR, exist_ok=True)
    os.makedirs(SHOT_MISSES_CLIPS_DIR, exist_ok=True); os.makedirs(TURNOVERS_CLIPS_DIR, exist_ok=True)

    producer_thread = threading.Thread(target=process_video_file_thread, args=(SOURCE_VIDEO_FILE,), name="VideoChunker", daemon=True)
    producer_thread.start()
    for i in range(NUM_WORKER_THREADS):
        worker = threading.Thread(target=clip_processing_worker, name=f"ClipProcessor-{i+1}", daemon=True)
        worker.start()
    logging.info(f"Started {NUM_WORKER_THREADS} worker threads for hybrid analysis.")

if __name__ == '__main__':
    init_thread = threading.Thread(target=start_background_tasks, daemon=True)
    init_thread.start()
    try:
        app.run(host='0.0.0.0', port=5002, threaded=True)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught, signaling threads to stop...")
        stop_event.set()
