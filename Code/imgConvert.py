import sys
import os
from PIL import Image

### Using pillow==10.3.0

def print_usage():
    print("Usage: python imgConvert.py <source_path> <target_path> <width> <height>")
    print("Example: python imgConvert.py ./source_images ./target_images 800 600")
    sys.exit(1)

def convert_images(source_path, target_path, width, height):
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist")
        sys.exit(1)

    # Create target directory if it doesn't exist
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Get all files in source directory (non-recursive)
    files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

    # Process each file
    for filename in files:
        # Check if file is jpg or png
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        source_file = os.path.join(source_path, filename)
        target_file = os.path.join(target_path, filename)

        # Check if target file already exists
        if os.path.exists(target_file):
            print(f"Skipping '{filename}' - already exists in target directory")
            continue

        try:
            # Open and resize image
            with Image.open(source_file) as img:
                try:
                    # Try newer PIL version method
                    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fall back to older PIL version method
                    resized_img = img.resize((width, height), Image.ANTIALIAS)
                resized_img.save(target_file)
            print(f"Converted '{filename}' successfully")
        except Exception as e:
            print(f"Error processing '{filename}': {str(e)}")

def main():
    # Check arguments
    if len(sys.argv) != 5:
        print_usage()

    try:
        source_path = sys.argv[1]
        target_path = sys.argv[2]
        width = int(sys.argv[3])
        height = int(sys.argv[4])
    except ValueError:
        print("Error: Width and height must be integers")
        print_usage()

    convert_images(source_path, target_path, width, height)

if __name__ == "__main__":
    main()
