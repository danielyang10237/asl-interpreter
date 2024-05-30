import os
import shutil

source_dir = 'images'
dest_dir = 'augmented_asl_dataset'

subdirectories_images = os.listdir(source_dir)

for subdirectory in subdirectories_images:
    # Construct the full path to the source subdirectory
    source_subdir_path = os.path.join(source_dir, subdirectory)

    # Determine the corresponding destination subdirectory name (convert letters to lowercase)
    dest_subdir_name = subdirectory.lower()
    
    # Construct the full path to the destination subdirectory
    dest_subdir_path = os.path.join(dest_dir, dest_subdir_name)

    # Ensure the destination subdirectory exists (optional, remove if you are sure they exist)
    if not os.path.exists(dest_subdir_path):
        os.makedirs(dest_subdir_path)

    # check if source_subdir_path is a directory
    if not os.path.isdir(source_subdir_path):
        continue

    # Move all files from the source subdirectory to the corresponding destination subdirectory
    for file_name in os.listdir(source_subdir_path):
        source_file_path = os.path.join(source_subdir_path, file_name)
        dest_file_path = os.path.join(dest_subdir_path, file_name)
        
        # Move file (use shutil.move to move and replace existing files if necessary)
        shutil.move(source_file_path, dest_file_path)

print("Files moved successfully.")
