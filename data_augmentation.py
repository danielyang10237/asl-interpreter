import os
import cv2
import numpy as np

def add_random_background(image):
    if image is None:
        print("Error: Image not found.")
        return
    
    background_color = np.array([0, 0, 0])
    upper_bound = np.array([5, 5, 5])
    
    # Create a mask where the background pixels are
    mask = cv2.inRange(image, background_color, upper_bound)
    
    # Generate random RGB values
    random_background = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    
    # Apply the random background only where the mask is true
    image[mask == 255] = random_background[mask == 255]
    
    return image

def random_scale(image):
    if image is None:
        print("Error: Image not found.")
        return

    # Generate a random scale factor between 0.5 and 1.5
    scale_factor = np.random.uniform(0.2, 0.99)

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Resize the image
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding to add to reach original dimensions
    top_padding = (original_height - new_height) // 2
    bottom_padding = original_height - new_height - top_padding
    left_padding = (original_width - new_width) // 2
    right_padding = original_width - new_width - left_padding

    # print(top_padding, bottom_padding, left_padding, right_padding)

    # create an image of original dimension 
    padded_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def random_stretch(image):
    if image is None:
        print("Error: Image not found.")
        return
    
    original_height, original_width = image.shape[:2]
    
    # Generate a random stretch factor for the horizontal axis
    stretch_x = np.random.uniform(0.8, 1.2)  # Stretch between 80% and 120%
    
    # Resize the image horizontally
    new_width = int(original_width * stretch_x)
    resized_image = cv2.resize(image, (new_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    if new_width > original_width:
        # Crop the image if it's wider than the original
        start_x = (new_width - original_width) // 2
        stretched_image = resized_image[:, start_x:start_x + original_width]
    else:
        # Add padding if the image is narrower than the original
        padding_left = (original_width - new_width) // 2
        padding_right = original_width - new_width - padding_left
        stretched_image = cv2.copyMakeBorder(resized_image, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return stretched_image

def random_rotate(image):
    if image is None:
        print("Error: Image not found.")
        return
    
    # Generate a random angle between -30 and 30 degrees
    angle = np.random.uniform(-30, 30)
    
    # Get the image center
    center = (image.shape[1] / 2, image.shape[0] / 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Rotate the image
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    return image

def random_flip(image):
    if image is None:
        print("Error: Image not found.")
        return
    
    # Generate a random number between 0 and 1
    flip = np.random.randint(0, 2)
    
    # Flip the image horizontally
    if flip == 1:
        image = cv2.flip(image, 1)
    
    return image

def random_translate(image, scale_factor):
    if image is None:
        print("Error: Image not found.")
        return
    
    # Generate random translation values
    x = np.random.randint(-scale_factor * image.shape[1], scale_factor * image.shape[1])
    y = np.random.randint(-scale_factor * image.shape[0], scale_factor * image.shape[0])
    
    # Get the translation matrix
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    
    # Translate the image
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    return image

def transform_image(image_path, destination_folder, background=True, scale=True, rotate=True, flip=True, stretch=True, translate=True, tag=""):
    image = cv2.imread(image_path)
    
    if scale:
        image = random_scale(image)

    if stretch:
        image = random_stretch(image)
    
    if rotate:
        image = random_rotate(image)
    
    if flip:
        image = random_flip(image)
    
    if translate:
        image = random_translate(image, 0.1)

    if background:
        image = add_random_background(image)
    
    image_name = os.path.basename(image_path)
    image_name = image_name.split(".")[0] + tag + ".jpeg"
    destination_path = os.path.join(destination_folder, image_name)
    cv2.imwrite(destination_path, image)

# transform_image("asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg", "augmented_asl_dataset/0")

def main():
    data_path = "asl_dataset/"
    destination_folder = "augmented_asl_dataset/"
    
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        
        print("performing data augmentation for ", label)

        destination_label_path = os.path.join(destination_folder, label)
        if not os.path.exists(destination_label_path):
            os.makedirs(destination_label_path)
        
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)

            transform_image(img_path, destination_label_path, tag="_1")
            transform_image(img_path, destination_label_path, tag="_2")
            transform_image(img_path, destination_label_path, tag="_3")

if __name__ == '__main__':
    # transform_image("asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg", "augmented_asl_dataset/0")
    main()