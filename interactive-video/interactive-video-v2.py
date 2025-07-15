import cv2
import numpy as np
import pygame
import time
from moviepy.editor import VideoFileClip
from scenedetect import SceneManager, ContentDetector, open_video
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialize BLIP model for scene descriptions
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to detect key scenes
def detect_scenes(video_path, max_scenes=10):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) > max_scenes:
        step = len(scene_list) // max_scenes
        scene_list = [scene_list[i] for i in range(0, len(scene_list), step)][:max_scenes]

    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

# Function to describe scene using BLIP model
def describe_scene(frame_path):
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Extract a frame and describe it
def extract_frame(video_path, timestamp, frame_path="scene_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(frame_path, frame)
        return describe_scene(frame_path)
    return "No visual description available"

# Extract and save audio from video
def extract_audio(video_path, audio_path="extracted_audio.mp3"):
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(audio_path, codec="mp3")
            return audio_path
        return None
    except Exception as e:
        print("Error extracting audio:", e)
        return None

# Function to display video and descriptions together
def play_video_with_audio(video_path, scenes, scene_summaries):
    audio_path = extract_audio(video_path)
    if not audio_path:
        print("Error: No audio found.")
        return

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    cap = cv2.VideoCapture(video_path)

    # **Proper resolution for video**
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Player", 1280, 720)  # Larger video window

    # **Description Panel (Below Video)**
    cv2.namedWindow("Scene Descriptions", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scene Descriptions", 800, 300)  # Fixed height

    current_scene_index = 0  # Start at scene 0
    cap.set(cv2.CAP_PROP_POS_MSEC, scenes[current_scene_index][0] * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Player", frame)

        # **Update Scene Descriptions**
        desc_window = np.zeros((300, 800, 3), dtype=np.uint8)  # Black background
        y_offset = 20
        for i, (start, end) in enumerate(scenes):
            text = f"[{i+1}] {scene_summaries.get(i+1, 'No description')}"
            cv2.putText(desc_window, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # **Small Font**
            y_offset += 20
        cv2.putText(desc_window, "[Q] Quit", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Scene Descriptions", desc_window)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break  # Quit the player

        # Handle scene selection (1 to max_scenes)
        for i in range(len(scenes)):
            if key == ord(str(i + 1)[-1]):
                # Jump to the selected scene **without closing video**
                current_scene_index = i
                cap.set(cv2.CAP_PROP_POS_MSEC, scenes[i][0] * 1000)
                pygame.mixer.music.play(start=scenes[i][0])
                break

    cap.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()

# Main function
def main():
    video_path = "enya.mp4"  # Update with your actual video file

    print("Detecting scenes...")
    scenes = detect_scenes(video_path, max_scenes=10)
    print(f"Detected {len(scenes)} key scenes.")

    scene_summaries = {i + 1: extract_frame(video_path, start) for i, (start, end) in enumerate(scenes)}

    print("Starting interactive video player...")
    play_video_with_audio(video_path, scenes, scene_summaries)

if __name__ == "__main__":
    main()
