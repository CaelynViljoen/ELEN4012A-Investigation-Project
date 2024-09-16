import os
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Defining the Dataset class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + ".png")
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Defining the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == '__main__': 

    # Define the directories and CSV file paths for the dataset to extract features from
    images_directory = "Data/APTOS-2019 Dataset/preprocessed_train_images"
    csv_file = "Data/APTOS-2019 Dataset/binary_train.csv"
    
    # Creating the dataset and dataloader
    dataset = FundusDataset(csv_file=csv_file, img_dir=images_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load the pretrained ResNet-18 model from the .pth file
    model = models.resnet18(weights=None)  # Load ResNet-18 without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 1)  # Modify the final layer to match the saved model structure
    
    model_path = os.path.join("Models", "resnet18_fundus_binary.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load the trained weights
    
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Modify the model for feature extraction
    model.fc = nn.Identity()  # Replace the final classification layer with an identity layer
    
    # Feature extraction after loading the model
    model.eval()  # Set the model to evaluation mode
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Extract the 512 features
            features = model(inputs)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    # Save the features and labels
    np.save("resnet18_512_features.npy", features_array) 
    np.save("resnet18_labels.npy", labels_array)
    
    print(f"Features and labels saved to 'resnet18_512_features.npy' and 'resnet18_labels.npy'")
