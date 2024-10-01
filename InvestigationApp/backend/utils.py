import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

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