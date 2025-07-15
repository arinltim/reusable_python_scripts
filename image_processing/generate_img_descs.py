import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from PIL import Image

# Set device to CPU
device = torch.device("cpu")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.to(device)
model.eval()

def generate_captions(image_path, num_captions=5):
    """Generates captions for a given image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,           # Sampling for variety
            temperature=0.6,          # Lower = more deterministic
            num_beams=5,              # Balanced exploration-exploitation
            num_return_sequences=num_captions,
            max_length=40,
            top_k=100,                # Consider top 100 tokens at each step
            top_p=0.9,                # Nucleus sampling for diversity
            length_penalty=1.0,       # Neutral length handling
            repetition_penalty=1.1,   # Mild penalty to prevent repetition
        )

    # Decode and return generated captions
    captions = [processor.decode(seq, skip_special_tokens=True) for seq in output]
    return captions

def process_image_folder(folder_path, num_captions=5):
    """Processes images and prints captions."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nProcessing image: {image_file}")

        try:
            captions = generate_captions(image_path, num_captions)
            for i, caption in enumerate(captions, 1):
                print(f"Caption {i}: {caption}")
        except Exception as e:
            print(f"Error processing '{image_file}': {str(e)}")

if __name__ == "__main__":
    images_folder = "img_enya"  # Set to your image folder path
    process_image_folder(images_folder, num_captions=5)
