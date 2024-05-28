import torch
import os
import cv2
from transformers import ViTForImageClassification, ViTImageProcessor

model_name = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name)

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    return model

def load_test_data(video_path, frames_per_second=10):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video")
        return None
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)

    frames = []

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if len(frames) % frame_interval == 0:
            frames.append(frame)
    
    video.release()

    return frames


if __name__ == "__main__":
    model_path = "asl_model_austin"
    model = load_model(model_path)

    videos = []
    frames = load_test_data("test_videos/vid1.mp4")

    videos.append(frames)

    for videos_frames in videos:
        for frame in videos_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze()
            outputs = model(pixel_values)
            highest_prob = torch.argmax(outputs.logits)
            print(f"Predicted label for frame: {highest_prob}")
