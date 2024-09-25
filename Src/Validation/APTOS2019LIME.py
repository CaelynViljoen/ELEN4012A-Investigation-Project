import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Step 1: Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 2: Load the Fine-tuned ResNet-18 Model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classification layer
model.load_state_dict(torch.load("Models/resnet18_fundus_binary.pth", map_location=device))
model = model.to(device)
model.eval()

# Step 3: Define Image Transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 4: Load and Transform the Image
image_path = "Data/APTOS-2019 Dataset/val_images/00a8624548a9.png"
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Step 5: LIME Explainer
explainer = lime_image.LimeImageExplainer()

# Step 6: Define a function that LIME will use to make predictions
def predict_fn(images):
    images = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
    images = images.to(device)
    outputs = model(images)
    probabilities = torch.sigmoid(outputs).cpu().detach().numpy()
    return np.concatenate([1 - probabilities, probabilities], axis=1)

# Step 7: Explain the prediction for the image
explanation = explainer.explain_instance(
    np.array(image),  # The image to explain
    predict_fn,       # The prediction function
    top_labels=1,     # Explain the top prediction
    hide_color=0,     # Hide superpixels that are not in the explanation
    num_samples=1000  # Number of times to perturb the image
)

# Step 8: Get the image and mask that LIME produces
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],  # The top prediction's label
    positive_only=True,         # Show only the parts that positively contributed
    num_features=10,            # Number of superpixels to highlight
    hide_rest=False             # Hide the rest of the image or not
)

# Step 9: Plot the explanation
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis('off')

plt.show() 
 