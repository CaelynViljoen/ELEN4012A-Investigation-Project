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
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]  # Label in the second column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Defining the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor, no further normalization as it's done in preprocessing
])

# Function to train and evaluate the model with multiple epochs
def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=5):
    # Store metrics for the last epoch
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_aucs = []
    all_fprs = []
    all_tprs = []

    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Correct label shape for BCE loss
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        # Validation step
        model.eval()
        val_loss = 0.0
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
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        epoch_aucs.append(roc_auc)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, AUC: {roc_auc:.4f}")

    # Return metrics from the last epoch of training
    return epoch_train_losses[-1], epoch_val_losses[-1], epoch_aucs[-1], all_fprs[-1], all_tprs[-1]

if __name__ == '__main__':
    images_directory = "Data/APTOS-2019 Dataset/preprocessed_train_images"
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
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)  # Binary classification with a single output
        
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train and evaluate the model on the current fold
        train_loss, val_loss, roc_auc, fpr, tpr = train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device)
        fold_metrics.append((train_loss, val_loss, roc_auc))
        
        print(f"Fold {fold + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {roc_auc:.4f}")
        
        # Save the model for this fold
        model_save_path = f"resnet18_fundus_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for Fold {fold + 1} to {model_save_path}")
        
        # Check if this is the best model based on AUC score
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_wts = model.state_dict()  # Save the best model's weights
        
        # Interpolate the TPR to the common FPR points and store
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure the curve starts at (0,0)
        
        # Plot the ROC curve for the current fold
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold+1} (area = {roc_auc:.2f})')
    
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
    
    # Display average performance across folds
    avg_train_loss = sum([x[0] for x in fold_metrics]) / len(fold_metrics)
    avg_val_loss = sum([x[1] for x in fold_metrics]) / len(fold_metrics)
    avg_auc = sum([x[2] for x in fold_metrics]) / len(fold_metrics)
    
    print(f"Average Performance - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, AUC: {avg_auc:.4f}")
    
    # Save the best model based on AUC
    if best_model_wts:
        best_model_save_path = "resnet18_fundus_best.pth"
        torch.save(best_model_wts, best_model_save_path)
        print(f"Best model saved to {best_model_save_path} with AUC: {best_auc:.4f}")
