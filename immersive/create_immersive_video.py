import os
import ssl
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from moviepy.editor import VideoFileClip

# Set GitHub token for torch.hub authentication (if needed)
os.environ["GITHUB_TOKEN"] = ""

# Disable SSL verification (for environments with SSL issues)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# Load MiDaS model from torch.hub with error handling
def load_model():
    try:
        device = torch.device("cpu")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.to(device).eval()
        print("[INFO] MiDaS model loaded successfully.")
        return model
    except Exception as e:
        print("[ERROR] Failed to load MiDaS model:", e)
        exit(1)

# Convert video to mp4 format (if needed)
def convert_to_mp4(input_path):
    if not input_path.endswith(".mp4"):
        output_path = input_path.rsplit(".", 1)[0] + ".mp4"
        print(f"[INFO] Converting {input_path} to {output_path}...")
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec="libx264")
        clip.close()
        return output_path
    return input_path

# Process video and create immersive output
def create_immersive_video(input_video_path, output_video_path):
    model = load_model()
    device = torch.device("cpu")

    # Ensure video is in MP4 format
    input_video_path = convert_to_mp4(input_video_path)

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print("[ERROR] Failed to open video file.")
        exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video Info: {frame_width}x{frame_height} at {fps} FPS, {total_frames} frames")

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Preprocessing transformations for MiDaS_small model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame_count = 0
    while frame_count < total_frames:
        if frame_count > 500:  # Stop after 500 frames
            break
        ret, frame = video.read()
        if not ret:
            print("[INFO] End of video or frame read error.")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"[INFO] Processing frame {frame_count}/{total_frames}")

        # Convert frame to tensor
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Run depth estimation
        with torch.no_grad():
            depth_map = model(input_tensor)

        # Normalize depth map
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Resize depth map to match frame size
        depth_map_resized = cv2.resize((depth_map * 255).astype(np.uint8), (frame_width, frame_height))

        # Apply color map for visualization
        depth_map_colored = cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)

        # Blend original frame with depth map
        blended_frame = cv2.addWeighted(frame, 0.7, depth_map_colored, 0.3, 0)

        # Write the blended frame to output video
        out.write(blended_frame)

        # Clean up to release memory
        del input_tensor, depth_map, depth_map_resized, depth_map_colored
        torch.cuda.empty_cache()

    # Clean up resources
    video.release()
    out.release()
    print("[INFO] Immersive video creation complete:", output_video_path)

if __name__ == "__main__":
    input_video = "enya.flv"  # Input video file (FLV or MP4)
    output_video = "immersive_video.mp4"  # Output immersive video

    create_immersive_video(input_video, output_video)
