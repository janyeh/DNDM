#!/usr/bin/env python3
import os
import sys
import argparse
import random
from PIL import Image

# Helper function to filter image files by common extensions
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])

def main():
    parser = argparse.ArgumentParser(description="Randomly crop images from source directory until target count is reached.")
    parser.add_argument("source_path", type=str, help="Path to the source images directory")
    parser.add_argument("target_path", type=str, help="Path to save cropped images")
    parser.add_argument("total_amount", type=int, help="Total number of cropped images to generate")
    parser.add_argument("width", type=int, help="Width of the cropped image")
    parser.add_argument("height", type=int, help="Height of the cropped image")
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path
    total_amount = args.total_amount
    crop_width = args.width
    crop_height = args.height

    # Check if source_path exists and is a directory
    if not os.path.isdir(source_path):
        print(f"Error: Source path '{source_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Create target_path if it doesn't exist
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
            print(f"Created target directory: {target_path}")
        except Exception as e:
            print(f"Error creating target directory '{target_path}': {e}")
            sys.exit(1)

    # Gather all image files from source_path
    image_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if is_image_file(f)]
    if not image_files:
        print(f"Error: No image files found in source path '{source_path}'.")
        sys.exit(1)

    saved_count = 0
    file_index = 0
    num_source = len(image_files)

    while saved_count < total_amount:
        # Cycle through the image list
        current_image_path = image_files[file_index % num_source]
        file_index += 1

        try:
            with Image.open(current_image_path) as img:
                orig_width, orig_height = img.size
                # Check if the image is large enough for the desired crop
                if orig_width < crop_width or orig_height < crop_height:
                    print(f"Skipping '{current_image_path}' because its size ({orig_width}x{orig_height}) is smaller than the crop size ({crop_width}x{crop_height}).")
                    continue

                # Choose a random top-left coordinate for the crop
                max_x = orig_width - crop_width
                max_y = orig_height - crop_height
                left = random.randint(0, max_x)
                upper = random.randint(0, max_y)
                right = left + crop_width
                lower = upper + crop_height

                cropped_img = img.crop((left, upper, right, lower))
                
                # Save the cropped image with a sequential filename
                output_filename = os.path.join(target_path, f"crop_{saved_count:04d}.jpg")
                cropped_img.save(output_filename)
                print(f"Saved cropped image: {output_filename}")
                saved_count += 1

        except Exception as e:
            print(f"Error processing file '{current_image_path}': {e}")
            continue

    print(f"Finished generating {saved_count} cropped images.")

if __name__ == "__main__":
    main()
