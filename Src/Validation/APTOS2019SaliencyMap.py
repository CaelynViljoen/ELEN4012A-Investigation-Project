import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Fine-tuned ResNet-18 Model
model = models.resnet18(weights=None)  # Start with an untrained ResNet-18
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classification layer

# Load the saved weights from fine-tuned model
#model.load_state_dict(torch.load("Models/resnet18_fundus_binary.pth")) #---------------------------------------------
model.load_state_dict(torch.load("Models/resnet18_best_resizeWithAspectRatioKept.pth"))

# Set the model to evaluation mode (important for inference)
model.eval()

# Step 2: Resize and Pad the Original Image
def resize_and_pad_image(img):
    # Resize the image while preserving the aspect ratio
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

# Load the original image
original_image_path = "Data/APTOS-2019 Dataset/val_images/0a61bddab956.png"
original_img = Image.open(original_image_path)

#preprocessed_image_path = "Data/APTOS-2019 Dataset/preprocessed_val_images/02dda30d3acf.png" #-----------------
#preprocessed_img = Image.open(preprocessed_image_path) #-------------------------------------------------------

# Resize and pad the image
resized_original_img = resize_and_pad_image(original_img)
preprocessed_img = resize_and_pad_image(original_img) #--------------------------------------------------------

# Convert the resized image to a tensor without normalizing (for model input)
transform = transforms.ToTensor()
img_tensor = transform(preprocessed_img).unsqueeze(0)  # Add batch dimension

# Ensure the image tensor requires gradients (necessary for saliency maps)
img_tensor.requires_grad_()

# Step 3: Forward Pass and Get Prediction
output = model(img_tensor) # Forward pass: Get the model's output

# Apply sigmoid to convert output to probabilities
probability = torch.sigmoid(output).item()

# Determine the predicted class based on the probability
predicted_class = "Diabetic" if probability > 0.5 else "Not Diabetic"

# Zero out previous gradients and backpropagate to get gradients with respect to the input image
model.zero_grad()
output.backward()

# Step 4: Generate the Saliency Map
saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1) # Get the gradients of the input image

# Reshape the saliency map to the image size
saliency = saliency.reshape(224, 224).numpy()

# Step 5: Visualize the Resized Image, Saliency Map, and Prediction
plt.figure(figsize=(12, 6)) # Plot the resized and padded original image
plt.subplot(1, 4, 1)
plt.imshow(resized_original_img)
plt.title("Resized Original Image", fontsize=14, pad=100)
plt.axis('off')

# Plot the saliency map
plt.subplot(1, 4, 2)
plt.imshow(saliency, cmap=plt.cm.hot)
plt.title("Saliency Map", fontsize=14, pad=100)
plt.axis('off')

# Plot the preprocessed image
plt.subplot(1, 4, 3)
plt.imshow(preprocessed_img)
plt.title("Preprocessed Image", fontsize=14, pad=100)
plt.axis('off')

# Display the prediction and class identification 
plt.subplot(1, 4, 4)
plt.text(0.5, 0.6, predicted_class, fontsize=16, ha='center', va='center')  # Display the prediction status
plt.text(0.5, 0.4, f"Class Identification = {probability:.2f}", fontsize=16, ha='center', va='center')  # Display the probability
plt.title("Model Prediction", fontsize=14)
plt.axis('off')

plt.show()