import os
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Function to resize and pad the image while preserving the aspect ratio
def resize_and_pad_image(img, desired_size=224):
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)

    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_img

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
        image = Image.open(img_name).convert("RGB")
        image = resize_and_pad_image(image, desired_size=224)
        label = self.data_frame.iloc[idx, 1]  # Assuming labels are already integers from 0 to 4

        if self.transform:
            image = self.transform(image)

        return image, label

# Defining the image transformations: Only ToTensor as resizing is handled in the dataset class
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to train and evaluate the model with multiple epochs
def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted_val = torch.max(outputs, 1)
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Main script
if __name__ == '__main__':
    images_directory = "Data/APTOS-2019 Dataset/train_images"
    csv_file = "Data/APTOS-2019 Dataset/train.csv"
    
    # Creating the dataset
    dataset = FundusDataset(csv_file=csv_file, img_dir=images_directory, transform=transform)
    skf = StratifiedKFold(n_splits=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Store metrics for each fold
    fold_metrics = []
    best_model_wts = None
    best_val_accuracy = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.data_frame.iloc[:, 1])):
        print(f"Fold {fold + 1}")
        
        # Create data loaders for the current fold
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, num_workers=4)
        
        # Load the pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # Multiclass classification with five outputs
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train and evaluate the model on the current fold
        train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
            train_loader, val_loader, model, criterion, optimizer, device
        )
        fold_metrics.append((train_losses, val_losses, train_accuracies, val_accuracies))
        
        print(f"Fold {fold + 1} - Best Validation Accuracy: {max(val_accuracies):.4f}%")
        
        # Save the best model weights based on validation accuracy
        if max(val_accuracies) > best_val_accuracy:
            best_val_accuracy = max(val_accuracies)
            best_model_wts = model.state_dict()
    
    # Save the best model based on validation accuracy
    if best_model_wts:
        best_model_save_path = "multiclass_resnet.pth"
        torch.save(best_model_wts, best_model_save_path)
        print(f"Best model saved to {best_model_save_path} with Validation Accuracy: {best_val_accuracy:.4f}%")

    # Plotting accuracy and loss across epochs
    for fold_num, metrics in enumerate(fold_metrics, start=1):
        train_losses, val_losses, train_accuracies, val_accuracies = metrics
        
        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label=f'Fold {fold_num} Train Accuracy')
        plt.plot(val_accuracies, label=f'Fold {fold_num} Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Training and Validation Accuracy for Fold {fold_num}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label=f'Fold {fold_num} Train Loss')
        plt.plot(val_losses, label=f'Fold {fold_num} Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for Fold {fold_num}')
        plt.legend()
        plt.grid(True)
        plt.show()