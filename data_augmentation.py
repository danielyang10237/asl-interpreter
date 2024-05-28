import os
import cv2
import numpy as np

def add_random_background(image_path, target_folder):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found.")
        return
    
    # Define the background color threshold (assumes pure black)
    # You may need to adjust the threshold values if the background isn't pure black
    background_color = np.array([0, 0, 0])
    upper_bound = np.array([5, 5, 5])
    
    # Create a mask where the background pixels are
    mask = cv2.inRange(image, background_color, upper_bound)
    
    # Generate random RGB values
    random_background = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    
    # Apply the random background only where the mask is true
    image[mask == 255] = random_background[mask == 255]
    
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Save the image with a random background
    new_image_path = os.path.join(target_folder, os.path.basename(image_path))
    cv2.imwrite(new_image_path, image)
    print(f"Image saved to {new_image_path}")

add_random_background("asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg", "augmented_asl_dataset/0")
