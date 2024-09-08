"""
This script contains the preprocessing steps applied to the APTOS2019 Dataset. The images are resized to match the necessary size 
requirements for the ResNet-18 and GoogleNet models. The images are resized and padded with zeros to ensure that no information is lost,
regardless of the size of the original images. The images are standardised from 0-255 to 0-1 and then normalised using the mean and 
standard deviations of the ImageNet dataset (as this is what the CNNs were originally trained on).
"""
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def resize_with_padding(image, target_size=(224, 224)):
    """Resize image to target size while maintaining aspect ratio and adding padding if necessary."""
    original_size = image.size  # Original size (width, height)
    
    # Calculate the ratio of target size to original size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    
    # Compute new size with the same aspect ratio
    new_size = tuple([int(x * ratio) for x in original_size])
    
    # Resize the image
    image = image.resize(new_size, Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality downsampling
    
    # Create a new image with the target size and a black background
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    
    # Paste the resized image onto the center of the new image
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2,
                            (target_size[1] - new_size[1]) // 2))
    
    return new_image

# Define the transformations including the custom resize function
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, target_size=(224, 224))),
    transforms.ToTensor(),  # Converts the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes using ImageNet stats
])

def preprocess_and_display_image(image_path):
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")
    
    # Apply the transformations to get the resized image
    resized_image = resize_with_padding(original_image, target_size=(224, 224))
    
    # Display the original and resized images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Resized Image")
    plt.imshow(resized_image)
    plt.axis('off')
    
    plt.show()
    
    # Apply the remaining transformations (tensor conversion and normalization)
    image_tensor = transform(resized_image)
    
    return image_tensor

# Example usage
image_path = 'C:/Users/Kaylin/OneDrive/Documents/Fourth Year IE 2024/Investigation Project/ELEN4012A-Investigation-Project/Data/APTOS-2019 Dataset/train_images/1ae8c165fd53.png'
image_tensor = preprocess_and_display_image(image_path)

# To verify the tensor size
print("Processed image tensor shape:", image_tensor.shape)



"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def resize_with_padding(image, target_size=(224, 224)):
    Resizes image to target size while maintaining aspect ratio and adding padding if necessary.
    original_size = image.size  # Original size (width, height)
    
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1]) # Calculates the ratio of target size to original size
    new_size = tuple([int(x * ratio) for x in original_size]) # Computes the new size with the same aspect ratio
    image = image.resize(new_size, Image.ANTIALIAS) # Resizes the image
    
    new_image = Image.new("RGB", target_size, (0, 0, 0)) # Creates a new image with the target size and a black background
    
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2,
                            (target_size[1] - new_size[1]) // 2)) # Pastes the resized image onto the center of the new image
    
    return new_image

# Defines the transformations including the custom resize function
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, target_size=(224, 224))),
    transforms.ToTensor(),  # Converts the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalises using ImageNet stats
])

def preprocess_image(image_path):
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image) # Applys the transformations
    
    return image

# Example usage
image_path = 'C:/Users/Kaylin/OneDrive/Documents/Fourth Year IE 2024/Investigation Project/ELEN4012A-Investigation-Project/Data/APTOS-2019 Dataset/train_images/1ae8c165fd53.png'
image_tensor = preprocess_image(image_path)

# To verify the tensor size
print("Processed image tensor shape:", image_tensor.shape)
"""