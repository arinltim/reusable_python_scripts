# app.py (MODIFIED to handle stream discontinuities)

import os
import shutil
import threading
import queue
import time
import json
import logging
from datetime import datetime
import ffmpeg

from flask import Flask, render_template, jsonify, send_from_directory, abort, Response, request
import google.generativeai as genai
import whisper
from streamlink import Streamlink
import werkzeug.utils
import certifi
import urllib3 # [NEW] Import urllib3 to manage warnings

# [NEW] Silence the InsecureRequestWarning messages from the logs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION & FLASK APP SETUP (Unchanged) ---
CLIP_DURATION_SECONDS = 20
OUTPUT_CLIPS_DIR = "output_clips_hls"
CATEGORIZED_CLIPS_DIR = os.path.join("static", "categorized_clips_hls")
NUM_WORKER_THREADS = 2
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')

# --- GLOBAL VARIABLES & THREADING (Unchanged) ---
stop_event = threading.Event()
clip_queue = queue.Queue()
sse_queue = queue.Queue()
captured_clips_info = []
captured_clips_lock = threading.Lock()
WHISPER_MODEL = None
cutter_thread_instance = None
cutter_thread_lock = threading.Lock()

current_session_id = None
worker_threads = []

# --- Helper functions like transcription, gemini_analysis etc. are unchanged ---
def transcription(model, clip_path: str, clip_filename: str) -> str:
    logging.info(f"→ Transcribing video from: {clip_path}...")
    try:
        result = model.transcribe(clip_path, fp16=False); text = result["text"]
        logging.info(f"✔ Transcription successful for: {clip_filename}"); return text
    except Exception as e:
        logging.error(f"❗ Error transcribing {clip_filename} with Whisper: {e}"); return ""


def gemini_analysis(text_content: str, clip_filename: str) -> str:
    """
    Analyzes the transcribed text using Gemini's JSON Mode for guaranteed
    valid JSON output.
    """
    genai.configure(api_key="AIzaSyDUngYlgwM2HVLl4XQW1Px6-jRxDLwnEFU") # Replace with your key

    # 1. [NEW] Configure the model to use JSON mode.
    # This guarantees the model's entire response is a valid JSON string.
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    # 2. [NEW] The prompt can be simplified as the JSON structure is now enforced by the API.
    prompt = f"""
    Analyze the following news transcript. Your entire response must be a single JSON object.

    The JSON object must contain two keys:
    1. "category": A specific category for the news content. Choose from: "Breaking News", "War & Conflict", "Weather", "Politics", "Business & Finance", "Sports", or create a new specific one like "Technology" or "Health". Use "General News" as a fallback for generic content.
    2. "summary": A concise, one-sentence summary of the clip's content.

    ---
    Transcript:
    "{text_content}"
    ---
    """
    try:
        response = model.generate_content(prompt)

        # 3. [MODIFIED] No more string cleaning is needed. The response text is the JSON.
        logging.info(f"--- Gemini Analysis for {clip_filename} ---\n{response.text}\n---------------------------------")
        return response.text

    except Exception as e:
        logging.error(f"❗ Error during Gemini analysis for {clip_filename}: {e}", exc_info=True)
        # Return a valid JSON error object
        return json.dumps({"category": "Error", "summary": f"Failed to analyze text: {e}"})


def categorize_and_move_clip(video_path: str, thumb_path: str, gemini_output_str: str) -> (str, str, str, str):
    try:
        analysis = json.loads(gemini_output_str); raw_category = analysis.get("category", "General News").strip(); category = raw_category.replace(" & ", "_and_").replace(" ", "_"); summary = analysis.get("summary", "No summary provided.")
    except (json.JSONDecodeError, AttributeError):
        raw_category = "General News"; category = "General_News"; summary = "Could not parse analysis."
    destination_dir = os.path.join(CATEGORIZED_CLIPS_DIR, category); os.makedirs(destination_dir, exist_ok=True)
    video_filename = os.path.basename(video_path); thumb_filename = os.path.basename(thumb_path)
    try:
        shutil.move(video_path, os.path.join(destination_dir, video_filename)); shutil.move(thumb_path, os.path.join(destination_dir, thumb_filename))
        logging.info(f"Moved '{video_filename}' and '{thumb_filename}' to '{destination_dir}'")
        video_web_path = f"{category}/{video_filename}"; thumb_web_path = f"{category}/{thumb_filename}"
        return video_web_path, thumb_web_path, raw_category, summary
    except Exception as e:
        logging.error(f"Error moving files for '{video_filename}': {e}"); return None, None, None, None

def live_stream_cutter_thread(real_stream_url: str, session_id: str):
    logging.info(f"✅ [CUTTER_THREAD] ({session_id}): Thread has started.")
    try:
        session = Streamlink()
        # These options improve compatibility with different streams
        session.set_option("http-ssl-verify", False)
        session.set_option("http-headers", {"User-Agent": "Mozilla/5.0"})
        session.set_option("hls-live-edge", 2)
        session.set_option("hls-timeout", 30)

        streams = session.streams(real_stream_url)
        if 'best' not in streams:
            logging.error(f"❌ [CUTTER_THREAD] ({session_id}): 'best' stream not found."); return

        with streams['best'].open() as fd:
            logging.info(f"✅ [CUTTER_THREAD] ({session_id}): Stream opened. Entering capture loop.")
            clip_counter = 0
            while not stop_event.is_set():
                clip_path = os.path.join(OUTPUT_CLIPS_DIR, f"clip_{session_id}_{clip_counter}.ts")

                try:
                    with open(clip_path, 'wb') as outfile:
                        # This logic is simpler and more robust for reading a segment
                        start_time = time.time()
                        while time.time() - start_time < CLIP_DURATION_SECONDS:
                            if stop_event.is_set(): break
                            data = fd.read(8192)
                            if not data:
                                logging.warning(f"❌ [CUTTER_THREAD] ({session_id}): Stream ended."); stop_event.set(); break
                            outfile.write(data)

                    if stop_event.is_set():
                        if os.path.exists(clip_path): os.remove(clip_path)
                        break

                    # Put the path AND the session ID on the queue
                    clip_queue.put((clip_path, session_id))
                    logging.info(f"✅ [CUTTER_THREAD] ({session_id}): Enqueued clip {clip_counter}"); clip_counter += 1

                except Exception as read_e:
                    logging.error(f"❌ [CUTTER_THREAD] ({session_id}): Error during stream read/write: {read_e}")
                    if os.path.exists(clip_path): os.remove(clip_path) # Clean up partial file
                    break # Exit loop on error
    except Exception as e:
        logging.error(f"❌ [CUTTER_THREAD] ({session_id}): A critical exception occurred: {e}", exc_info=True)
    finally:
        stop_event.set()
        logging.info(f"✅ [CUTTER_THREAD] ({session_id}): Thread is finishing.")

def clip_processing_worker():
    """This version correctly unpacks the tuple from the queue."""
    while True:
        try:
            # --- [FIX] Correctly unpack the two items from the queue ---
            ts_clip_path, work_session_id = clip_queue.get(timeout=1)

            # This block is now wrapped in its own try/except to handle processing errors
            # without killing the worker thread.
            try:
                mp4_clip_path = convert_ts_to_mp4(ts_clip_path)
                if not mp4_clip_path:
                    raise Exception("MP4 conversion returned None")

                thumbnail_path = extract_thumbnail(mp4_clip_path)
                if not thumbnail_path:
                    raise Exception("Thumbnail extraction returned None")

                clip_filename = os.path.basename(mp4_clip_path)
                transcript_text = transcription(WHISPER_MODEL, mp4_clip_path, clip_filename)

                if transcript_text:
                    gemini_output = gemini_analysis(transcript_text, clip_filename)

                    with cutter_thread_lock:
                        # Check if the session is still active before publishing
                        if work_session_id == current_session_id:
                            video_web_path, thumb_web_path, category, summary = categorize_and_move_clip(mp4_clip_path, thumbnail_path, gemini_output)
                            if video_web_path:
                                clip_data = {"video_path": video_web_path, "thumb_path": thumb_web_path, "category": category, "summary": summary, "timestamp": datetime.now().strftime('%H:%M:%S')}
                                sse_queue.put(clip_data)
                                with captured_clips_lock: captured_clips_info.append(clip_data)
                        else:
                            logging.warning(f"Discarding stale result from previous session '{work_session_id}'.")
                            if os.path.exists(mp4_clip_path): os.remove(mp4_clip_path)
                            if os.path.exists(thumbnail_path): os.remove(thumbnail_path)
                else: # No transcript found
                    if os.path.exists(mp4_clip_path): os.remove(mp4_clip_path)
                    if os.path.exists(thumbnail_path): os.remove(thumbnail_path)

            except Exception as e:
                logging.error(f"An error occurred during clip processing for '{os.path.basename(ts_clip_path)}': {e}")
            finally:
                # Always clean up the original .ts chunk and mark task done
                if os.path.exists(ts_clip_path):
                    os.remove(ts_clip_path)
                clip_queue.task_done()

        except queue.Empty:
            # This is normal. The queue was empty. The loop will continue.
            if stop_event.is_set():
                logging.info(f"[{threading.current_thread().name}] Stop event set and queue empty. Exiting.")
                break # Exit the perpetual loop only when stop is signaled AND queue is empty
            continue


def convert_ts_to_mp4(ts_path: str) -> str | None:
    """
    Converts a .ts video file to .mp4 using ffmpeg.
    This version re-encodes the stream to fix any discontinuities and preserve audio.
    """
    mp4_path = ts_path.replace(".ts", ".mp4")
    logging.info(f"→ Converting '{os.path.basename(ts_path)}' to MP4 (re-encoding)...")
    try:
        ffmpeg.input(ts_path).output(
            mp4_path,
            vcodec='libx264',
            acodec='aac',
            # [NEW] Add audio bitrate and strictness level to ensure audio track is created
            audio_bitrate='192k',
            strict='experimental'  # Often needed for AAC codec
        ).run(overwrite_output=True, quiet=True)

        logging.info(f"✔ Conversion successful: '{os.path.basename(mp4_path)}'")
        return mp4_path
    except ffmpeg.Error as e:
        logging.error(f"❗ FFmpeg error during conversion of {ts_path}", exc_info=True)
        return None

def extract_thumbnail(mp4_path: str) -> str | None:
    thumbnail_path = mp4_path.replace(".mp4", ".jpg"); logging.info(f"→ Extracting thumbnail for '{os.path.basename(mp4_path)}'...")
    try:
        ffmpeg.input(mp4_path, ss=1).output(thumbnail_path, vframes=1).run(overwrite_output=True, quiet=True)
        logging.info(f"✔ Thumbnail extracted: '{os.path.basename(thumbnail_path)}'"); return thumbnail_path
    except ffmpeg.Error as e:
        logging.error(f"❗ FFmpeg error during thumbnail extraction of {mp4_path}", exc_info=True); return None


# ==============================================================================
# SECTION 2: FLASK ROUTES (Unchanged)
# ==============================================================================
@app.route('/')
def index():
    return render_template('index_hls.html')

@app.route('/start-processing', methods=['POST'])
def start_processing():
    global cutter_thread_instance, worker_threads, current_session_id
    with cutter_thread_lock:
        if cutter_thread_instance and cutter_thread_instance.is_alive():
            return jsonify({"status": "error", "message": "A stream is already being processed."}), 409

        # --- [CHANGED] This block now fully initializes a new session ---
        stop_event.clear()
        session_id = str(time.time())
        current_session_id = session_id

        # Start new worker threads for this session
        worker_threads = []
        for i in range(NUM_WORKER_THREADS):
            worker = threading.Thread(target=clip_processing_worker, name=f"ClipProcessor-{i+1}", daemon=True)
            worker.start()
            worker_threads.append(worker)

        # Start the new cutter thread
        data = request.get_json()
        url = data.get('stream_url')
        if not url:
            return jsonify({"status": "error", "message": "stream_url not provided in request."}), 400

        cutter_thread_instance = threading.Thread(target=live_stream_cutter_thread, args=(url, session_id), name="StreamCutter", daemon=True)
        cutter_thread_instance.start()

        logging.info(f"Started new session {session_id} with {NUM_WORKER_THREADS} workers for URL: {url}")
        return jsonify({"status": "success", "message": "Backend processing started."})


@app.route('/stop-processing', methods=['POST'])
def stop_processing():
    global cutter_thread_instance, worker_threads, current_session_id
    with cutter_thread_lock:
        logging.info("Received request to stop processing.")

        # Signal all threads to begin shutdown
        stop_event.set()
        current_session_id = None

        # Wait for the cutter thread to finish
        if cutter_thread_instance and cutter_thread_instance.is_alive():
            logging.info("Waiting for StreamCutter thread to join...")
            cutter_thread_instance.join(timeout=5)

        # --- [CHANGED] Now also wait for all worker threads to finish ---
        logging.info("Waiting for worker threads to finish processing queue...")
        for worker in worker_threads:
            if worker.is_alive():
                worker.join(timeout=30) # Wait up to 30s per worker

        # Clear queues and reset state
        cutter_thread_instance = None
        worker_threads = []
        with captured_clips_lock:
            captured_clips_info.clear()
        while not clip_queue.empty():
            try: clip_queue.get_nowait()
            except queue.Empty: break

        # Clean up old files
        if os.path.exists(CATEGORIZED_CLIPS_DIR):
            shutil.rmtree(CATEGORIZED_CLIPS_DIR)
        os.makedirs(CATEGORIZED_CLIPS_DIR, exist_ok=True)

        logging.info("Backend processing stopped and state has been fully reset.")
        return jsonify({"status": "success", "message": "Processing stopped and state reset."})

@app.route('/initial-clips')
def initial_clips():
    with captured_clips_lock:
        return jsonify(captured_clips_info)

@app.route('/categorized_clips_hls/<category>/<filename>')
def serve_clip(category, filename):
    safe_category = werkzeug.utils.secure_filename(category)
    safe_filename = werkzeug.utils.secure_filename(filename)
    if safe_category != category or safe_filename != filename:
        abort(404)
    try:
        directory = os.path.join(os.getcwd(), "static", "categorized_clips_hls", safe_category)
        return send_from_directory(directory, safe_filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            try:
                data = sse_queue.get(timeout=1)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                if stop_event.is_set():
                    logging.info("[SSE_THREAD]: Stop event detected, closing stream.")
                    break
    return Response(event_stream(), mimetype='text/event-stream')

# ==============================================================================
# SECTION 3: APPLICATION STARTUP (Unchanged)
# ==============================================================================
def start_background_tasks():
    global WHISPER_MODEL
    # Clean up directories once on application startup
    if os.path.exists(CATEGORIZED_CLIPS_DIR): shutil.rmtree(CATEGORIZED_CLIPS_DIR)
    if os.path.exists(OUTPUT_CLIPS_DIR): shutil.rmtree(OUTPUT_CLIPS_DIR)
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(CATEGORIZED_CLIPS_DIR, exist_ok=True)

    logging.info("Loading Whisper 'small' model...")
    try:
        WHISPER_MODEL = whisper.load_model("small")
        logging.info("✔ Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Could not load Whisper model: {e}"); return

    logging.info("Application initialized. Waiting for stream URL from frontend...")

if __name__ == '__main__':
    init_thread = threading.Thread(target=start_background_tasks, daemon=True)
    init_thread.start()
    app.run(host='0.0.0.0', port=5001, threaded=True)