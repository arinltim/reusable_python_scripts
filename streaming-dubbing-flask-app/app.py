import os
import shutil
import threading
import queue
import time
import logging
import io
import asyncio
import ssl
from functools import partial
import json
import base64

from flask import Flask, render_template, jsonify, Response, request
import google.generativeai as genai
import whisper
from streamlink import Streamlink
import urllib3
import edge_tts
import nltk

# --- Configuration & Setup ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
AUDIO_CHUNKS_DIR = "audio_chunks"
CHUNK_DURATION_SECONDS = 4
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s')

# --- Global State & Threading ---
stop_event = threading.Event()
raw_audio_queue = queue.Queue(maxsize=5)
active_translation_queues = []
translation_queues_lock = threading.Lock()
dubbed_data_queues = {}
dubbing_threads = {}
dubbing_threads_lock = threading.Lock()
WHISPER_MODEL = None
GEMINI_MODEL = None
stream_cutter_thread = None
transcription_thread = None

# --- Core Processing Functions ---

async def translate_and_speak_sentence(loop: asyncio.AbstractEventLoop, sentence: str, language_name: str, edge_voice: str, output_queue: queue.Queue):
    """
    Translates a sentence using a sync call and synthesizes it asynchronously.
    """
    try:
        # Step 1: Translate the sentence
        prompt = f"Translate the following English text to {language_name}. Provide ONLY the translated text, no pre-amble, no formatting, just the raw translated text:\n\n'{sentence}'"

        # [MODIFIED] Using functools.partial for a cleaner and more robust call
        blocking_call = partial(GEMINI_MODEL.generate_content, prompt)
        response = await loop.run_in_executor(None, blocking_call)

        translated_text = response.text.strip().replace("*", "")
        if not translated_text:
            return

        logging.info(f"Sub-task: Translated '{sentence}' -> '{translated_text}'")

        # Step 2: Synthesize the translated sentence
        audio_buffer = io.BytesIO()
        communicate = edge_tts.Communicate(translated_text, edge_voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            logging.warning(f"Edge-TTS produced no audio for: '{translated_text}'")
            return

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        output_queue.put({"text": translated_text, "audio_base64": audio_base64})
        logging.info(f"Sub-task: Finished synthesis for '{translated_text}'")
    except Exception as e:
        # This will now clearly log if the Gemini API call fails
        logging.error(f"CRITICAL: Error during translate/speak task for sentence '{sentence}': {e}", exc_info=True)


async def process_text_block(loop: asyncio.AbstractEventLoop, text_block: str, language_name: str, edge_voice: str, output_queue: queue.Queue):
    sentences = nltk.sent_tokenize(text_block)
    logging.info(f"Split text into {len(sentences)} sentences. Processing concurrently.")
    tasks = [
        translate_and_speak_sentence(loop, sentence, language_name, edge_voice, output_queue)
        for sentence in sentences if sentence.strip()
    ]
    await asyncio.gather(*tasks)
    logging.info("Finished processing all sentences in the block.")


def text_translator_and_dubber(target_lang: str, input_queue: queue.Queue):
    # [ADDED] A guard to ensure the model is ready before starting
    if not GEMINI_MODEL:
        logging.error(f"Dubber-{target_lang}: Cannot start, Gemini model is not available.")
        return

    logging.info(f"Dubber-{target_lang}: Thread started.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    voice_map = {"es-ES": "es-ES-ElviraNeural", "de-DE": "de-DE-KatjaNeural", "fr-FR": "fr-FR-DeniseNeural", "hi-IN": "hi-IN-SwaraNeural"}
    edge_voice = voice_map.get(target_lang)
    language_name = {'es-ES': 'Spanish', 'de-DE': 'German', 'fr-FR': 'French', 'hi-IN': 'Hindi'}.get(target_lang, '')

    if not edge_voice:
        logging.error(f"Dubber-{target_lang}: No voice configured."); return

    try:
        while not stop_event.is_set():
            try:
                original_text_block = input_queue.get(timeout=1)
                logging.info(f"Dubber-{target_lang}: Received text block to process.")
                loop.run_until_complete(process_text_block(
                    loop, original_text_block, language_name, edge_voice, dubbed_data_queues[target_lang]
                ))
                input_queue.task_done()
            except queue.Empty: continue
            except Exception as e:
                logging.error(f"Dubber-{target_lang}: Error in main loop: {e}", exc_info=True)
                time.sleep(2)
    finally:
        logging.info(f"Dubber-{target_lang}: Closing event loop.")
        loop.close()
    logging.info(f"Dubber-{target_lang}: Thread finishing.")


# --- Other functions and Flask routes are unchanged ---
def stream_audio_cutter(stream_url):
    logging.info("AudioCutter: Thread started.")
    session = Streamlink(); session.set_option("http-ssl-verify", False); session.set_option("hls-live-edge", 3); session.set_option("hls-timeout", 20)
    try:
        streams = session.streams(stream_url)
        if 'best' not in streams: logging.error("AudioCutter: 'best' stream not found."); return
        with streams['best'].open() as stream_fd:
            logging.info("AudioCutter: Stream opened.")
            while not stop_event.is_set():
                chunk_data = io.BytesIO(); start_time = time.time()
                try:
                    while time.time() - start_time < CHUNK_DURATION_SECONDS:
                        if stop_event.is_set(): break
                        data = stream_fd.read(1024 * 8)
                        if not data: logging.warning("AudioCutter: Stream ended."); stop_event.set(); break
                        chunk_data.write(data)
                    if stop_event.is_set(): break
                    chunk_data.seek(0); raw_audio_queue.put(chunk_data)
                except Exception as e: logging.error(f"AudioCutter: Error reading stream: {e}", exc_info=True); time.sleep(2)
    except Exception as e: logging.error(f"AudioCutter: Critical error: {e}", exc_info=True)
    finally: logging.info("AudioCutter: Thread finishing."); stop_event.set()

def audio_transcriber():
    logging.info("Transcriber: Thread started.")
    while not stop_event.is_set() or not raw_audio_queue.empty():
        try:
            audio_chunk_bytes = raw_audio_queue.get(timeout=1)
            temp_file_path = os.path.join(AUDIO_CHUNKS_DIR, f"temp_{time.time_ns()}.mp3")
            with open(temp_file_path, 'wb') as f: f.write(audio_chunk_bytes.read())
            result = WHISPER_MODEL.transcribe(temp_file_path, fp16=False)
            text = result["text"]; os.remove(temp_file_path)
            if text and text.strip():
                logging.info(f"Transcriber: Got text: '{text}'")
                with translation_queues_lock:
                    for q in active_translation_queues: q.put(text)
            raw_audio_queue.task_done()
        except queue.Empty: continue
        except Exception as e: logging.error(f"Transcriber: Error: {e}", exc_info=True)
    logging.info("Transcriber: Thread finishing.")

@app.route('/')
def index(): return render_template('index.html')
@app.route('/dubbing-stream')
def dubbing_stream():
    lang = request.args.get('lang')
    if lang not in dubbed_data_queues: return "Language not activated or supported", 404
    def generate_events():
        logging.info(f"SSE-Stream-{lang}: Client connected.")
        while not stop_event.is_set():
            try:
                data_bundle = dubbed_data_queues[lang].get(timeout=1)
                yield f"data: {json.dumps(data_bundle)}\n\n"
                dubbed_data_queues[lang].task_done()
            except queue.Empty: continue
            except Exception as e: logging.error(f"SSE-Stream-{lang}: Error: {e}"); break
        logging.info(f"SSE-Stream-{lang}: Client disconnected.")
    return Response(generate_events(), mimetype='text/event-stream')
@app.route('/start-dubbing', methods=['POST'])
def start_dubbing():
    global stream_cutter_thread, transcription_thread
    if stream_cutter_thread and stream_cutter_thread.is_alive(): return jsonify({"status": "error", "message": "A stream is already active."}), 409
    data = request.get_json(); stream_url = data.get('stream_url')
    if not stream_url: return jsonify({"status": "error", "message": "stream_url is required."}), 400
    stop_event.clear()
    with translation_queues_lock: active_translation_queues.clear()
    with dubbing_threads_lock: dubbing_threads.clear()
    for q in list(dubbed_data_queues.values()):
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
    if os.path.exists(AUDIO_CHUNKS_DIR): shutil.rmtree(AUDIO_CHUNKS_DIR)
    os.makedirs(AUDIO_CHUNKS_DIR)
    logging.info(f"Starting pipeline for URL: {stream_url}")
    stream_cutter_thread = threading.Thread(target=stream_audio_cutter, args=(stream_url,), name="AudioCutter", daemon=True); stream_cutter_thread.start()
    transcription_thread = threading.Thread(target=audio_transcriber, name="Transcriber", daemon=True); transcription_thread.start()
    return jsonify({"status": "success", "message": "Transcription pipeline started."})
@app.route('/activate-language', methods=['POST'])
def activate_language():
    lang = request.args.get('lang')
    if not lang: return jsonify({"status": "error", "message": "lang parameter is required."}), 400
    with dubbing_threads_lock:
        if lang not in dubbing_threads:
            logging.info(f"Activating dubbing for language: {lang}")
            input_queue = queue.Queue(); dubbed_data_queues[lang] = queue.Queue(maxsize=50)
            with translation_queues_lock: active_translation_queues.append(input_queue)
            thread = threading.Thread(target=text_translator_and_dubber, args=(lang, input_queue), name=f"Dubber-{lang}", daemon=True); thread.start()
            dubbing_threads[lang] = thread
        else:
            logging.info(f"Dubbing for language '{lang}' is already active.")
    return jsonify({"status": "success", "message": f"Dubbing for {lang} activated."})
@app.route('/stop-dubbing', methods=['POST'])
def stop_dubbing():
    logging.info("Received request to stop all processing.")
    stop_event.set()
    threads_to_join = [stream_cutter_thread, transcription_thread]
    with dubbing_threads_lock: threads_to_join.extend(dubbing_threads.values())
    for thread in threads_to_join:
        if thread and thread.is_alive(): thread.join(timeout=5)
    logging.info("All threads have been stopped.")
    return jsonify({"status": "success", "message": "Processing stopped."})

# --- Application Startup ---
def initialize_app():
    global WHISPER_MODEL, GEMINI_MODEL
    logging.info("Initializing application...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # [MODIFIED] Stricter check at startup
        logging.error("FATAL: GEMINI_API_KEY environment variable not set. Translation will fail. Please set it and restart.")
        # We don't exit, but we set the model to None so threads will fail gracefully.
        GEMINI_MODEL = None
    else:
        try:
            genai.configure(api_key=api_key)
            GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
            logging.info("Gemini (translation) client initialized successfully.")
        except Exception as e:
            logging.error(f"FATAL: Could not initialize Gemini client: {e}", exc_info=True)
            GEMINI_MODEL = None # Ensure model is None on failure

    try:
        logging.info("Downloading NLTK sentence tokenizer data (punkt)...")
        nltk.download('punkt', quiet=True)
        logging.info("NLTK data downloaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Could not download NLTK data: {e}", exc_info=True); exit()

    try:
        logging.info("Loading Whisper 'tiny.en' model...")
        WHISPER_MODEL = whisper.load_model("tiny.en")
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL: Could not load Whisper model: {e}", exc_info=True); exit()

    if os.path.exists(AUDIO_CHUNKS_DIR): shutil.rmtree(AUDIO_CHUNKS_DIR)
    os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
    logging.info("Application initialization complete.")


if __name__ == '__main__':
    logging.warning(
        "MONKEY-PATCHING SSL: Replacing ssl.create_default_context with "
        "ssl._create_unverified_context. This is insecure and for testing only."
    )
    ssl.create_default_context = ssl._create_unverified_context
    initialize_app()
    logging.info("Starting Flask server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True)

