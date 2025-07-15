import os
import base64
import tempfile
import subprocess
from flask import Flask, request, render_template, jsonify
from google.cloud import speech
import imageio_ffmpeg as ffmpeg

# Initialize Flask app
app = Flask(__name__)

# Set your Google Cloud credentials (update with your JSON key file path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service-account.json"

# Initialize the Google Cloud Speech client
client = speech.SpeechClient()

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receives base64-encoded audio data from the frontend, converts the WebM audio
    to WAV (mono, 16KHz) using ffmpeg (via imageio-ffmpeg), and transcribes it.
    """
    try:
        data = request.get_json()
        audio_data = data.get("audio")
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400

        # Remove header (e.g., "data:audio/webm;base64,") and decode the audio bytes
        header, encoded = audio_data.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        # Write the WebM audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input.flush()
            input_filename = temp_input.name

        # Create a temporary file for the output WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            output_filename = temp_output.name

        # Get the bundled ffmpeg executable (from imageio-ffmpeg)
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        # Run ffmpeg command to convert the WebM file to WAV (mono, 16KHz)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", input_filename,
            "-ac", "1",        # force mono audio
            "-ar", "16000",    # force 16KHz sample rate
            output_filename
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_message = result.stderr.decode()
            return jsonify({"error": f"Error converting audio: {error_message}"}), 500

        with open(output_filename, "rb") as f:
            wav_bytes = f.read()

        # Clean up temporary files
        os.remove(input_filename)
        os.remove(output_filename)

        # Prepare the audio for Google Speech API
        speech_audio = speech.RecognitionAudio(content=wav_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        # Transcribe the audio using synchronous recognition
        response = client.recognize(config=config, audio=speech_audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
