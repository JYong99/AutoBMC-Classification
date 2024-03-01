import os
import shutil
import random

def split_images(input_folder, output_folder_prefix, num_folders):
    # Create output folders
    output_folders = [f"{output_folder_prefix}_{i}" for i in range(1, num_folders + 1)]
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Shuffle the image files
    random.shuffle(image_files)

    # Calculate the number of images to put in each folder
    images_per_folder = len(image_files) // num_folders

    # Distribute the images into the output folders
    for i in range(num_folders):
        start_idx = i * images_per_folder
        end_idx = (i + 1) * images_per_folder if i < num_folders - 1 else None
        current_folder = output_folders[i]

        for image in image_files[start_idx:end_idx]:
            image_path = os.path.join(input_folder, image)
            shutil.copy(image_path, current_folder)

if __name__ == "__main__":
    input_folder = "/home/dxd_jy/joel/Capstone/Dataset/Reject/Error/ErrorFile"
    output_folder_prefix = "/home/dxd_jy/joel/Capstone/Dataset/Reject/Error/Temp"
    num_folders = 10

    split_images(input_folder, output_folder_prefix, num_folders)
