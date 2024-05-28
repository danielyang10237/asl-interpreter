import torch
import os
import cv2
import csv
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

model_name = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name)

test_images_path = "personal_data"

def load_csv_as_dict(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        result_dict = {rows[1]: rows[0] for rows in reader}
    return result_dict

labels_dict = load_csv_as_dict('labels.csv')

def process_photo(photo_path):
    img = cv2.imread(photo_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = Image.fromarray(img)

    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].squeeze()
    return pixel_values

if __name__ == "__main__":
    # Load the Hugging Face model from the specified directory
    model_path = "asl_model_austin"  # Directory where the model with config.json and model.safetensors is saved
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()

    # Loop through all jpeg images in the personal_data folder
    for img in os.listdir(test_images_path):
        if img.endswith(".jpeg"):
            img_path = os.path.join(test_images_path, img)
            pixel_values = process_photo(img_path)
            outputs = model(pixel_values.unsqueeze(0))  # Add batch dimension
            highest_prob = torch.argmax(outputs.logits)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            top_five_predictions = torch.topk(probabilities, 5)
            print("RESULTS FOR", img_path)
            for i in range(5):
                print(f"Top {i + 1} prediction: {labels_dict[str(top_five_predictions.indices[0][i].item())]} with probability {top_five_predictions.values[0][i].item()}")
