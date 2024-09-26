import os
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage import color

# Function to resize, pad the image while preserving the aspect ratio, apply black background and denoising
def preprocess_image(img, desired_size=224):
    """
    Preprocess the image by resizing it while keeping the aspect ratio, padding it with black background,
    and applying denoising and texture extraction.

    Parameters:
        img (PIL.Image): The input image.
        desired_size (int): The size to which the image will be padded (default is 224).

    Returns:
        np.array: The preprocessed image ready for LBP texture extraction.
    """
    # Convert the image to RGB format if not already
    img = img.convert("RGB")
    
    # Resizing the image while preserving the aspect ratio
    old_size = img.size  # old_size is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create a new black image with the desired size and paste the resized image onto the center
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    
    # Convert the image to a numpy array for further processing
    image_np = np.array(new_img)

    # Convert to grayscale to detect the fundus boundary
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Threshold the grayscale image to create a binary mask
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Find the contours of the fundus region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the largest contour (assumed to be the fundus) on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Apply the mask to remove the background in the original image
    image_with_black_bg = cv2.bitwise_and(image_np, image_np, mask=mask)
    
    # Apply Non-Local Means Denoising for effective noise reduction
    denoised_image = cv2.fastNlMeansDenoisingColored(image_with_black_bg, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Convert the denoised image to grayscale and then to uint8 for LBP
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    
    # Compute the Local Binary Pattern (LBP)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Convert LBP to float32 to match PyTorch expectations
    lbp = lbp.astype(np.float32)

    # Stack the single channel LBP into 3 channels to match the input requirements of the model
    lbp_image = np.stack([lbp] * 3, axis=-1)
    
    return lbp_image

# Defining the Dataset class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + ".png")  # Image file name in the first column
        image = Image.open(img_name)  # Load the image
        
        # Preprocess the image: resize, pad, apply black background, denoise, and extract LBP texture
        preprocessed_image = preprocess_image(image, desired_size=224)
        
        # Convert to a tensor if a transform is provided
        if self.transform:
            preprocessed_image = self.transform(preprocessed_image)
        
        label = self.data_frame.iloc[idx, 1]  # Label in the second column
        
        return preprocessed_image, label

# Defining the image transformations: Only ToTensor as resizing and LBP are handled in the dataset class
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

# Function to train and evaluate the model with multiple epochs
def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=5):
    # Store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    epoch_aucs = []
    all_fprs = []
    all_tprs = []

    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Correct label shape for BCE loss
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted_train = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        val_accuracies.append(train_accuracy)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()  # Probability of the positive class
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                predicted_val = (probs > 0.5).astype(float)
                correct_val += (predicted_val == labels.cpu().numpy()).sum()
                total_val += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        epoch_aucs.append(roc_auc)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, AUC: {roc_auc:.4f}")

    # Return metrics from the last epoch of training
    return train_losses, val_losses, val_accuracies, epoch_aucs, all_fprs, all_tprs

# Main script
if __name__ == '__main__':
    images_directory = "Data/APTOS-2019 Dataset/train_images"
    csv_file = "Data/APTOS-2019 Dataset/binary_train.csv"
    
    # Creating the dataset
    dataset = FundusDataset(csv_file=csv_file, img_dir=images_directory, transform=transform)
    skf = StratifiedKFold(n_splits=5)  # 5-fold cross-validation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Store metrics and ROC data for each fold
    fold_metrics = []
    best_model_wts = None
    best_auc = 0.0  # Track the best AUC score to save the best model
    mean_fpr = np.linspace(0, 1, 100)  # Common FPR points for averaging
    tprs = []  # Store TPRs for each fold
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.data_frame.iloc[:, 1])):
        print(f"Fold {fold + 1}")
        
        # Create data loaders for the current fold
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, num_workers=4)
        
        # Load the pre-trained ResNet-18 model
        model = models.resnet18(weights="IMAGENET1K_V1")  # Updated weights argument
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)  # Binary classification with a single output
        
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train and evaluate the model on the current fold
        train_losses, val_losses, val_accuracies, epoch_aucs, fpr, tpr = train_and_evaluate(
            train_loader, val_loader, model, criterion, optimizer, device
        )
        fold_metrics.append((train_losses, val_losses, val_accuracies, epoch_aucs))
        
        print(f"Fold {fold + 1} - Best AUC: {max(epoch_aucs):.4f}")
        
        # Save the best model weights based on AUC score
        if max(epoch_aucs) > best_auc:
            best_auc = max(epoch_aucs)
            best_model_wts = model.state_dict()
            best_fpr = fpr[-1]
            best_tpr = tpr[-1]
        
        # Interpolate the TPR to the common FPR points and store
        tprs.append(np.interp(mean_fpr, best_fpr, best_tpr))
        tprs[-1][0] = 0.0  # Ensure the curve starts at (0,0)
        
        # Plot the ROC curve for the current fold
        plt.plot(best_fpr, best_tpr, lw=1, alpha=0.3, label=f'ROC fold {fold+1} (area = {max(epoch_aucs):.2f})')
    
    # Plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1,1)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (area = {mean_auc:.2f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.2, label='± 1 std. dev.')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve Across Folds')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plotting accuracy over epochs
    for fold_num, metrics in enumerate(fold_metrics, start=1):
        plt.plot(metrics[2], label=f'Fold {fold_num} Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Across Folds')
    plt.legend()
    plt.show()
    
    # Save the best model based on AUC
    if best_model_wts:
        best_model_save_path = "resnet18_best_textureImagesInput.pth"
        torch.save(best_model_wts, best_model_save_path)
        print(f"Best model saved to {best_model_save_path} with AUC: {best_auc:.4f}")