import os
from PIL import Image

# Your resize_and_pad_image function
def resize_and_pad_image(img, desired_size=500):
    # Calculates the new size while preserving the aspect ratio
    old_size = img.size  # old_size is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create a new image with the desired size and paste the resized image onto the center
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    return new_img

def preprocess_images(input_folder, output_folder, desired_size=500):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all image filenames from the input folder
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process all images
    for i, image_filename in enumerate(image_filenames):
        try:
            # Open the image
            img_path = os.path.join(input_folder, image_filename)
            img = Image.open(img_path)
            
            # Resize and pad the image
            processed_img = resize_and_pad_image(img, desired_size=desired_size)
            
            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, image_filename)
            processed_img.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {image_filename}: {e}")

if __name__ == "__main__":
    input_folder = r"Data\APTOS-2019 Dataset\train_images"
    output_folder = r"Data\APTOS-2019 Dataset\resized_train_images"
    preprocess_images(input_folder, output_folder, desired_size=500)
