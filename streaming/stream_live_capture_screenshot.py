import google.generativeai as genai
import time
from streamlink import Streamlink
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
import subprocess
from PIL import Image
import io
import os
from datetime import datetime
import threading
import queue # For thread-safe communication

# Configure Google Generative AI with your API key
genai.configure(api_key="AIzaSyA2YXUaYM3lwc0msb08ANZ-a8FbvpV--PE") # REPLACE WITH YOUR ACTUAL KEY
vision_model = genai.GenerativeModel("gemini-2.0-flash") # Or "gemini-1.5-pro" if you have access

# --- Your realstream and analyze_frame_with_gemini functions (no change needed) ---
def realstream(page_url: str, timeout: float = 15.0) -> (str, str):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--incognito")
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = webdriver.Chrome(options=options)
    print("Chrome options loaded.")
    driver.get(page_url)
    time.sleep(2)  # allow page and player to load

    try:
        btns = driver.find_elements(By.CSS_SELECTOR, "button.amp-pause-overlay")
        if btns:
            btns[0].click()
            print("Clicked play overlay.")
    except Exception:
        print("Play overlay not found or not clickable; proceeding.")

    time.sleep(3)
    video_el = driver.find_element(By.CSS_SELECTOR, "video.amp-media-element")
    blob_src = video_el.get_attribute("src")
    print("Blob URL:", blob_src)

    deadline = time.time() + timeout
    real_url = None
    while time.time() < deadline and not real_url:
        for entry in driver.get_log("performance"):
            try:
                msg = json.loads(entry["message"])["message"]
                if msg.get("method") == "Network.responseReceived":
                    res = msg.get("params", {}).get("response", {})
                    mime = res.get("mimeType", "")
                    url = res.get("url", "")
                    if "application/vnd.apple.mpegurl" in mime or mime.startswith("video/"):
                        real_url = url
                        break
            except json.JSONDecodeError:
                continue
        if not real_url:
            time.sleep(0.5)
    driver.quit()
    if not real_url:
        raise RuntimeError("Unable to detect real stream URL from performance logs.")
    print("Real stream URL detected:", real_url)
    return blob_src, real_url

def analyze_frame_with_gemini(image_data: bytes) -> bool:
    try:
        response = vision_model.generate_content([
            "Is there a weather map or weather information displayed in this image? Respond with 'YES' or 'NO' only.",
            {
                'mime_type': 'image/jpeg',
                'data': image_data
            }
        ])
        return "YES" in response.text.strip().upper()
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return False

# --- New Thread Functions ---

# Thread 1: Reads from Streamlink and writes to FFmpeg's stdin
def stream_reader_thread(streamlink_fd, ffmpeg_stdin_pipe, stop_event):
    print("[StreamReader] Starting...")
    try:
        while not stop_event.is_set():
            data = streamlink_fd.read(8192)
            if not data:
                print("[StreamReader] Streamlink stream ended.")
                break
            try:
                ffmpeg_stdin_pipe.write(data)
            except (BrokenPipeError, ValueError):
                print("[StreamReader] FFmpeg stdin pipe was closed, likely because FFmpeg exited.")
                break
            except Exception as e:
                # This error happens if we write to a pipe that was just terminated
                if not stop_event.is_set():
                    print(f"[StreamReader] Error writing to FFmpeg stdin: {e}")
                break
    except Exception as e:
        print(f"[StreamReader] Unexpected error in read loop: {e}")
    finally:
        # This is the crucial part: always try to close the pipe gracefully
        print("[StreamReader] Exiting loop. Closing FFmpeg stdin pipe to signal EOF.")
        if ffmpeg_stdin_pipe and not ffmpeg_stdin_pipe.closed:
            try:
                ffmpeg_stdin_pipe.close()
            except Exception as e:
                print(f"[StreamReader] Error closing FFmpeg stdin: {e}")
        stop_event.set() # Ensure other threads are signaled to stop
        print("[StreamReader] Exiting.")

# Thread 2: Reads from FFmpeg's stdout and processes frames
def ffmpeg_output_processor_thread(ffmpeg_stdout_pipe, output_dir, stop_event):
    print("[FFmpegProcessor] Starting...")
    buffer = b''
    frame_count = 0
    try:
        while not stop_event.is_set():
            # Use a non-blocking read or a read with a timeout
            # Polling stdout for data
            output_chunk = ffmpeg_stdout_pipe.read(4096)
            if not output_chunk:
                # If no data, check if FFmpeg process is still alive.
                # If it's dead and no data, we should stop.
                if stop_event.is_set(): # Stop if main loop told us to
                    break
                # Small sleep to prevent busy-waiting if no data
                time.sleep(0.01)
                continue # Continue loop to check for new data or stop_event

            buffer += output_chunk

            while True:
                start_index = buffer.find(b'\xff\xd8') # JPEG Start
                if start_index == -1:
                    break

                end_index = buffer.find(b'\xff\xd9', start_index + 2) # JPEG End
                if end_index == -1:
                    break

                jpeg_data = buffer[start_index : end_index + 2]
                buffer = buffer[end_index + 2:]

                if jpeg_data:
                    # print(f"[FFmpegProcessor] Detected JPEG of size {len(jpeg_data)} bytes.") # Too verbose
                    if analyze_frame_with_gemini(jpeg_data):
                        frame_count += 1
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(output_dir, f"weather_map_{timestamp}_{frame_count}.jpg")
                        with open(filename, 'wb') as f:
                            f.write(jpeg_data)
                        print(f"[FFmpegProcessor] Captured weather map screenshot: {filename}")
                else:
                    print("[FFmpegProcessor] Warning: Empty JPEG data extracted.")
    except Exception as e:
        print(f"[FFmpegProcessor] Unexpected error: {e}")
    finally:
        print("[FFmpegProcessor] Exiting.")
        stop_event.set() # Signal other threads to stop


def continuous_stream_and_capture(duration_sec: int, stream_url: str, output_dir: str = "weather_screenshots"):
    os.makedirs(output_dir, exist_ok=True)
    session = Streamlink()
    streams = session.streams(stream_url)
    if 'best' not in streams:
        raise RuntimeError(f"No playable streams found for {stream_url}")
    stream = streams['best']

    print(f"Opening streamlink session for {stream_url}...")
    fd = stream.open()

    ffmpeg_log_file = os.path.join(output_dir, "ffmpeg_debug_log.txt")
    print(f"FFmpeg stdout/stderr will be written to: {ffmpeg_log_file}")

    ffmpeg_process = None
    stop_event = threading.Event()

    try:
        with open(ffmpeg_log_file, 'w') as log_f:
            ffmpeg_cmd = [
                'ffmpeg', '-i', 'pipe:0', '-f', 'image2pipe', '-vcodec', 'mjpeg',
                '-q:v', '5', '-an', '-r', '0.1', '-loglevel', 'debug', 'pipe:1'
            ]

            print("Starting FFmpeg process...")
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_f, universal_newlines=False
            )

            reader_thread = threading.Thread(target=stream_reader_thread,
                                             args=(fd, ffmpeg_process.stdin, stop_event))
            processor_thread = threading.Thread(target=ffmpeg_output_processor_thread,
                                                args=(ffmpeg_process.stdout, output_dir, stop_event))

            reader_thread.start()
            processor_thread.start()

            print(f"Threads started. Capturing for {duration_sec} seconds...")
            # Wait for the specified duration, but also check if a thread has stopped prematurely
            start_time = time.time()
            while time.time() - start_time < duration_sec:
                if stop_event.is_set():
                    print("[Main] A thread signaled to stop prematurely.")
                    break
                time.sleep(0.5)

            if not stop_event.is_set():
                print(f"[Main] Reached main program duration limit ({duration_sec}s). Signaling threads to stop.")

    except KeyboardInterrupt:
        print("[Main] Ctrl+C detected. Signaling threads to stop.")
    except Exception as e:
        print(f"[Main] An error occurred in main loop: {e}")
    finally:
        print("[Main] Cleanup initiated. Signaling threads to stop.")
        stop_event.set()

        # Define threads with default values in case they fail to be assigned
        reader_thread = locals().get('reader_thread')
        processor_thread = locals().get('processor_thread')

        # 1. Wait for the reader thread to finish. It is responsible for closing FFmpeg's stdin.
        if reader_thread and reader_thread.is_alive():
            print("[Main] Waiting for reader_thread to finish...")
            reader_thread.join(timeout=10)

        # 2. Wait for the FFmpeg process to exit gracefully now that its input is closed.
        if ffmpeg_process and ffmpeg_process.poll() is None:
            print("[Main] Waiting for FFmpeg process to exit...")
            try:
                ffmpeg_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[Main] FFmpeg did not exit gracefully, terminating.")
                ffmpeg_process.terminate()
                # Give it a moment to terminate before killing
                try:
                    ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[Main] FFmpeg did not terminate, killing.")
                    ffmpeg_process.kill()

        # 3. Now wait for the processor thread. It will exit once the FFmpeg stdout pipe closes.
        if processor_thread and processor_thread.is_alive():
            print("[Main] Waiting for processor_thread to finish...")
            processor_thread.join(timeout=10)

        if fd:
            print("[Main] Closing Streamlink file descriptor.")
            fd.close()

        print("[Main] All cleanup complete. Exiting.")


if __name__ == '__main__':
    print("Initializing...")
    # url = "https://www.wusa9.com/watch"
    url = "https://www.kare11.com/watch"
    # url = "https://www.wcnc.com/watch"
    # url = "https://www.thv11.com/watch"
    # url = "https://www.9news.com/watch"

    try:
        _, real_stream_url = realstream(url, 15)
        print(f"Starting continuous capture from: {real_stream_url}")
        capture_duration_sec = 60
        continuous_stream_and_capture(capture_duration_sec, real_stream_url)
    except Exception as e:
        print(f"An error occurred during setup: {e}")
