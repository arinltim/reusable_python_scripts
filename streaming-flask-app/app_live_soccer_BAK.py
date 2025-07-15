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

# Add this import to silence the noisy but harmless SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- CONFIGURATION ---
CLIP_DURATION_SECONDS = 20
OUTPUT_CLIPS_DIR = "output_clips_live"
CATEGORIZED_CLIPS_DIR = "static/categorized_clips_live"
HASH_DIFFERENCE_THRESHOLD = 10
FRAMES_PER_CLIP = 8

# --- FLASK APP SETUP & GLOBALS ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')
stop_event = threading.Event()

# Use a simple list for polling instead of a complex queue
new_clips_list = []
new_clips_lock = threading.Lock()

processing_thread_instance = None
processing_thread_lock = threading.Lock()
processed_hashes = set()
hash_lock = threading.Lock()


# ==============================================================================
# SECTION 1: CORE PROCESSING LOGIC
# ==============================================================================

def gemini_frame_analysis(clip_path: str, image_paths: list) -> str | None:
    """
    [FINAL VERSION] This prompt combines "chain-of-thought" with multi-event
    detection and the strictest rules for the highest possible accuracy.
    """
    try:
        system_instruction = "You are a meticulous, world-class soccer referee and analyst AI. Your primary directive is factual accuracy. First, provide a step-by-step reasoning of what you see. Then, based ONLY on that reasoning, provide a final JSON object. Your entire response must be a single, valid JSON object containing a single key 'events', which holds a list of event objects."

        model = genai.GenerativeModel("gemini-1.5-pro-latest", system_instruction=system_instruction)

        image_parts = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    image_parts.append({"mime_type": "image/jpeg", "data": f.read()})

        if not image_parts: return None

        text_prompt = f"""
            Analyze the following sequence of {len(image_parts)} frames from a soccer match. Identify all significant events.

            **Step 1: Reasoning**
            Describe, step-by-step, what is happening in the frames. Note the trajectory of the ball, the actions of the players, and the final outcome of the play. Be factual and base your description only on what is visible.

            **Step 2: JSON Output**
            Based on your reasoning, create a JSON object with an "events" key, containing a list of event objects. For each event object, use the following keys:
            - "event": Classify the event. CRUCIAL RULES:
                - A "Goal" is ONLY when the ENTIRE ball fully crosses the goal line.
                - A "Save" is ONLY when the goalkeeper makes clear contact to stop a shot that was on target.
                - A "Shot Miss" is a shot that clearly goes wide, over, or hits the post and stays out.
            - "timestamp_in_clip": The time in seconds within the 20-second clip where the event occurred.
            - "summary": A one-sentence description of the event. THIS SUMMARY MUST BE FACTUALLY CONSISTENT with the other visuals in the clip. For example, if 'event' is 'Goal' and you can identify that the team celebrating is USA, then the summary MUST describe USA scoring.
            - "player_jersey_color": The jersey color of the key player.
            - "player_jersey_number": The jersey number (as a number). Use 0 if not visible.

            If no significant events are visible, return an empty list for the "events" key.
            """

        prompt_parts = [text_prompt] + image_parts
        response = model.generate_content(prompt_parts, request_options={"timeout": 120})

        if response.text:
            json_start_index = response.text.find('{')
            if json_start_index != -1:
                return response.text[json_start_index:].strip().replace("```", "")
        return None
    except Exception as e:
        logging.error(f"❗ An exception occurred during Gemini frame analysis for {os.path.basename(clip_path)}: {e}")
        return None


def stream_processor_thread(stream_url: str, headers: dict):
    """
    [FINAL VERSION] This thread creates precise, re-cut highlights for each
    event found by the AI.
    """
    logging.info(f"✅ [PROCESSOR]: Thread started for {stream_url}")

    session = Streamlink()
    session.set_option("http-headers", headers)
    session.set_option("hls-live-edge", 2)
    session.set_option("hls-timeout", 30)
    session.set_option("http-ssl-verify", False)

    try:
        streams = session.streams(stream_url)
        if 'best' not in streams:
            logging.error("❌ [PROCESSOR]: 'best' quality stream not found."); return

        with streams['best'].open() as fd:
            logging.info("✅ [PROCESSOR]: Stream opened. Entering main processing loop.")
            clip_counter = 0
            while not stop_event.is_set():
                ts_path, mp4_path, extracted_frames = None, None, []

                try:
                    ts_path = os.path.join(OUTPUT_CLIPS_DIR, f"live_chunk_{clip_counter}.ts")
                    logging.info(f"Capturing chunk {clip_counter}...")
                    with open(ts_path, 'wb') as outfile:
                        start_time = time.time()
                        while time.time() - start_time < CLIP_DURATION_SECONDS:
                            if stop_event.is_set(): break
                            data = fd.read(8192)
                            if not data:
                                logging.warning("❌ [PROCESSOR]: Stream ended."); stop_event.set(); break
                            outfile.write(data)

                    if stop_event.is_set(): break

                    logging.info(f"Analyzing chunk {clip_counter}...")
                    mp4_path = convert_ts_to_mp4(ts_path)
                    if not mp4_path: continue

                    extracted_frames = extract_frames_for_analysis(mp4_path)
                    if not extracted_frames: continue

                    gemini_output = gemini_frame_analysis(mp4_path, extracted_frames)

                    if gemini_output:
                        analysis_data = json.loads(re.sub(r',\s*([}\]])', r'\1', gemini_output))
                        events_found = analysis_data.get("events", [])

                        if not events_found:
                            logging.info(f"No significant events in chunk {clip_counter}.")
                        else:
                            for event_data in events_found:
                                highlight_path, thumbnail_path = None, None
                                event = event_data.get("event")
                                timestamp_in_clip = event_data.get("timestamp_in_clip")

                                if not event or timestamp_in_clip is None: continue

                                # --- [RE-CUTTING LOGIC RESTORED] ---
                                pre_action_seconds = 3
                                post_action_seconds = 25
                                highlight_start = max(0, timestamp_in_clip - pre_action_seconds)
                                highlight_duration = pre_action_seconds + post_action_seconds

                                highlight_filename = f"highlight_{event.replace(' ', '_')}_{int(time.time())}.mp4"
                                highlight_path = os.path.join(OUTPUT_CLIPS_DIR, highlight_filename)

                                logging.info(f"Event '{event}' found. Creating {highlight_duration}s highlight clip...")
                                ffmpeg.input(mp4_path, ss=highlight_start).output(highlight_path, t=highlight_duration, vcodec='libx264', acodec='aac').run(overwrite_output=True, quiet=True)
                                # -----------------------------------

                                thumbnail_path = extract_thumbnail(highlight_path)
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
                                    summary = event_data.get("summary", "No summary.")
                                    handle_highlight_clip(highlight_path, thumbnail_path, summary, event)

                except Exception as e:
                    logging.error(f"Error during processing loop for chunk {clip_counter}: {e}")
                finally:
                    if mp4_path and os.path.exists(mp4_path): os.remove(mp4_path)
                    for frame in extracted_frames:
                        if os.path.exists(frame): os.remove(frame)
                    if os.path.exists(ts_path): os.remove(ts_path)
                    clip_counter += 1
    except Exception as e:
        logging.error(f"❌ [PROCESSOR]: A critical exception occurred: {e}", exc_info=True)
    finally:
        stop_event.set()
        logging.info(f"✅ [PROCESSOR]: Thread is finishing.")


def convert_ts_to_mp4(ts_path: str) -> str | None:
    mp4_path = ts_path.replace(".ts", ".mp4")
    try:
        ffmpeg.input(ts_path).output(mp4_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', strict='experimental').run(overwrite_output=True, quiet=True)
        return mp4_path
    except ffmpeg.Error: return None

def extract_frames_for_analysis(mp4_path: str) -> list:
    output_dir = os.path.dirname(mp4_path)
    frame_paths = []
    try:
        probe = ffmpeg.probe(mp4_path)
        duration = float(probe['format']['duration'])
        interval = duration / (FRAMES_PER_CLIP + 1)
        for i in range(FRAMES_PER_CLIP):
            seek_time = interval * (i + 1)
            frame_path = os.path.join(output_dir, f"{os.path.basename(mp4_path)}_frame_{i}.jpg")
            ffmpeg.input(mp4_path, ss=seek_time).output(frame_path, vframes=1).run(overwrite_output=True, quiet=True)
            frame_paths.append(frame_path)
        return frame_paths
    except Exception: return []

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
def index(): return render_template('index_live_soccer.html')

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

