import os
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Defining the Dataset class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + ".png")  # Where the image file name is in the first column
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]  # Where the label is in the second column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Defining the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Only convert to tensor, no normalisation as it has already been done in preprocessing
])

if __name__ == '__main__': 

    train_images_directory = "Data/APTOS-2019 Dataset/preprocessed_train_images"
    train_csv_file = "Data/APTOS-2019 Dataset/binary_train.csv"
    
    val_images_directory = "Data/APTOS-2019 Dataset/preprocessed_val_images"
    val_csv_file = "Data/APTOS-2019 Dataset/binary_val.csv"
    
    # Creating the datasets and dataloaders
    train_dataset = FundusDataset(csv_file=train_csv_file, img_dir=train_images_directory, transform=transform)
    val_dataset = FundusDataset(csv_file=val_csv_file, img_dir=val_images_directory, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = models.resnet18(pretrained=True) # Loading the pretrained ResNet-18 model
    
    # Modifying the final layer to match the binary classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Replace the final layer with a single output neuron for binary classification
    
    # Define the loss function and optimiser
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimiser
    
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Lists to store loss and accuracy for each epoch
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # The training loop:
    num_epochs = 10  # NB: adjust if necessary 
    
    for epoch in range(num_epochs):
        model.train()  # Sets the model to training mode
        
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Makes sure labels are in the right format
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimise
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        average_train_loss = running_loss / len(train_dataloader)
        train_losses.append(average_train_loss)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_train_loss}")
        
        # Validation step after training the model for this epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate the accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        average_val_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Validation Loss: {average_val_loss}, Validation Accuracy: {val_accuracy}%")
    
    print("Training and validation complete.")
    
    # Save the trained model
    model_save_path = "resnet18_fundus_binary.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the training loss, validation loss, and validation accuracy
    metrics_save_path = "training_metrics.csv"
    metrics_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accuracies
    })
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Training loss, validation loss, and validation accuracy saved to {metrics_save_path}")