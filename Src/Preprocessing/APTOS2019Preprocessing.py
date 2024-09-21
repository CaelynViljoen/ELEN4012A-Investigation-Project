import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

images_directory = "Data/APTOS-2019 Dataset/train_images"

output_directory = "Data/APTOS-2019 Dataset/preprocessed_train_images" # Output directory for preprocessed images
os.makedirs(output_directory, exist_ok=True)

# The image transformation with normalisation is defined
transform = transforms.Compose([
    transforms.ToTensor(), # Normalises from 0-255 to 0-1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standardises using ImageNet mean and standard deviation (mean = 0, s.d =1)
                         std=[0.229, 0.224, 0.225])  # Performs the operation: Standardised value = (Original Val - Mean)/Standard Deviation
])

def resize_and_pad_image(img):
    # Calculates the new size while preserving the aspect ratio
    old_size = img.size  # old_size is in (width, height) format
    desired_size = 224
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create a new image with the desired size and paste the resized image onto the center
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    return new_img

# Processing each image
for image_name in tqdm(os.listdir(images_directory)):
    image_path = os.path.join(images_directory, image_name)
    
    with Image.open(image_path) as img:
        padded_img = resize_and_pad_image(img) # Resizes and pads the image

        img_transformed = transform(padded_img) # Apply the remaining transformations (normalize, convert to tensor)
        
        # Saving the transformed image
        output_path = os.path.join(output_directory, image_name) 
        img_transformed_pil = transforms.ToPILImage()(img_transformed)
        img_transformed_pil.save(output_path)