import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_saliency_map(image):

    # Define the model setup and load the trained weights
    model = models.resnet18(weights=None)  # Start with an untrained ResNet-18
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification layer

    # Define the correct path to your model weights
    trained_resnet_model = os.path.join('Models', 'resnet18_best_resizeWithAspectRatioKept.pth')
    
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
    transform = transforms.ToTensor()  # Convert the image to a tensor without normalization

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

# Example usage of the saliency map function
if __name__ == "__main__":
    # Load the image directly as a PIL Image object
    image_path = "Data/APTOS-2019 Dataset/val_images/0a61bddab956.png"  # Replace with the path to your image
    image = Image.open(image_path)

    # Generate the saliency map
    saliency_map, resized_image = generate_saliency_map(image)

    # Display the results
    plt.figure(figsize=(12, 6))
    
    # Plot the resized image
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image)
    plt.title("Resized Image")
    plt.axis('off')

    # Plot the saliency map
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap=plt.cm.hot)
    plt.title("Saliency Map")
    plt.axis('off')

    plt.show()
