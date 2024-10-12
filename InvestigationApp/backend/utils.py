#InvestigationApp\backend\utils.py
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os


# Function to resize and pad the image while preserving the aspect ratio
def resize_and_pad_image(img, desired_size=224):
    old_size = img.size  # old_size is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create a new image with the desired size and paste the resized image onto the center
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    return new_img

def check_guess(classification, guess):
    return classification.lower() == guess.lower()

def predict_diabetes_from_image(image):
    """
    Predicts whether a fundus image indicates diabetes.

    Parameters:
        image (PIL.Image.Image): The input image object.

    Returns:
        str: "Diabetic" or "Non-Diabetic" based on the model's prediction.
        float: Probability percentage of the image being diabetic.
    """

    # Update the path to match the new location of the model
    trained_resnet_model = os.path.join('model', 'resnet_binary_model.pth')

    # Verify that the model file exists
    if not os.path.isfile(trained_resnet_model):
        raise FileNotFoundError(f"Model file not found at {trained_resnet_model}. Please check the path.")

    # Load and prepare the model
    model = models.resnet18(weights=None)  # Use weights=None to avoid deprecation warning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Replace the final layer with the same configuration as before

    # Load the trained weights
    model.load_state_dict(torch.load(trained_resnet_model, map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
    ])

    # Function to resize and pad the image while preserving the aspect ratio
    def resize_and_pad_image(img, desired_size=224):
        """
        Resize the image while keeping the aspect ratio and pad it to the desired size.

        Parameters:
            img (PIL.Image.Image): The input image object.
            desired_size (int): The size to which the image will be padded (default is 224).

        Returns:
            PIL.Image.Image: The resized and padded image.
        """
        old_size = img.size  # old_size is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.LANCZOS)
        
        # Create a new image with the desired size and paste the resized image onto the center
        new_img = Image.new("RGB", (desired_size, desired_size))
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))
        return new_img

    # Preprocess the image
    image = resize_and_pad_image(image, desired_size=224)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the image to the device
    image = image.to(device)
    
    # Run the image through the model
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()  # Get the probability
    
    # Determine if the image is diabetic or not
    if prob > 0.5:
        return "Diabetic", prob * 100
    else:
        return "Non-Diabetic", (1 - prob) * 100

def generate_saliency_map(image):
    # Define the model setup and load the trained weights
    model = models.resnet18(weights=None)  # Start with an untrained ResNet-18
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification layer

    # Define the correct path to your model weights
    trained_resnet_model = os.path.join('model', 'resnet_binary_model.pth')
    
    # Verify that the model file exists
    if not os.path.isfile(trained_resnet_model):
        raise FileNotFoundError(f"Model file not found at {trained_resnet_model}. Please check the path.")

    # Load the trained weights
    model.load_state_dict(torch.load(trained_resnet_model, map_location=torch.device('cpu'), weights_only=True))

    # Set the model to evaluation mode
    model.eval()

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the image transformation
    transform = transforms.ToTensor()

    # Function to resize and pad the image while preserving the aspect ratio
    def resize_and_pad_image(img, desired_size=224):
        old_size = img.size  # old_size is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.LANCZOS)
        
        # Create a new image with the desired size and paste the resized image onto the center
        new_img = Image.new("RGB", (desired_size, desired_size))
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))
        return new_img

    # Resize and pad the image
    resized_img = resize_and_pad_image(image)

    # Convert the resized image to a tensor and add a batch dimension
    img_tensor = transform(resized_img).unsqueeze(0)
    
    # Ensure the tensor requires gradients (necessary for saliency maps)
    img_tensor.requires_grad_()

    # Move the tensor to the device
    img_tensor = img_tensor.to(device)

    # Forward pass to get the model output
    output = model(img_tensor)

    # Backward pass to get the gradients with respect to the input image
    model.zero_grad()
    output.backward()

    # Generate the saliency map
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()  # Convert to NumPy array for visualization

    return saliency, resized_img


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from PIL import Image

# Load and preprocess the image
def exudates_marked_image(image):
    
    # Function to resize and pad the image while preserving the aspect ratio
    def resize_and_pad_image(img, desired_size=224):
        # img is a NumPy array (OpenCV image), we use shape to get height and width
        old_size = img.shape[:2]  # old_size is in (height, width) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        
        # Resize the image using OpenCV
        img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a new image with the desired size and paste the resized image onto the center
        new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
        new_img[
            (desired_size - new_size[0]) // 2 : (desired_size - new_size[0]) // 2 + new_size[0],
            (desired_size - new_size[1]) // 2 : (desired_size - new_size[1]) // 2 + new_size[1]
        ] = img
        return new_img
    
   # image = resize_and_pad_image(image)

    # Convert to grayscale to detect the fundus boundary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ## REPLACE BLACK BACKGROUND CODE
    # Threshold the grayscale image to create a binary mask
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Find the contours of the fundus region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the largest contour (assumed to be the fundus) on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Apply the mask to remove the background in the original image
    image_with_black_bg = cv2.bitwise_and(image, image, mask=mask)

    hsv_image = cv2.cvtColor(image_with_black_bg, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for yellow in HSV (exudates are often yellowish)
    lower_yellow = np.array([20, 50, 50])  # Adjust these values as needed
    upper_yellow = np.array([40, 255, 255])
    
    # Create a binary mask where yellow regions are detected
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Apply morphological operations to remove noise and small false detections
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    # Label connected regions (exudates)
    labels = measure.label(yellow_mask, connectivity=2)
    regions = measure.regionprops(labels)

    # Function to mark detected exudates on the original image
    marked_image = image_with_black_bg.copy()

    # Draw circles around detected exudates
    for region in regions:
        # Get the centroid of each exudate region
        y, x = region.centroid
        
        # Draw a circle around the centroid on the original image
        cv2.circle(marked_image, (int(x), int(y)), 25, (0, 0, 0), 15)  # Blue circles

    # # Function to resize and pad the image while preserving the aspect ratio
    # def resize_and_pad_image(img, desired_size=224):
    #     # img is a NumPy array (OpenCV image), we use shape to get height and width
    #     old_size = img.shape[:2]  # old_size is in (height, width) format
    #     ratio = float(desired_size) / max(old_size)
    #     new_size = tuple([int(x * ratio) for x in old_size])
        
    #     # Resize the image using OpenCV
    #     img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
    #     # Create a new image with the desired size and paste the resized image onto the center
    #     new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    #     new_img[
    #         (desired_size - new_size[0]) // 2 : (desired_size - new_size[0]) // 2 + new_size[0],
    #         (desired_size - new_size[1]) // 2 : (desired_size - new_size[1]) // 2 + new_size[1]
    #     ] = img
    #     return new_img
    
    marked_image = resize_and_pad_image(marked_image)

    return marked_image
