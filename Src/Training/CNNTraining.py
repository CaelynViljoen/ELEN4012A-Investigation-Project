import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Input
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define the custom Dataset class
class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + ".png")  # Image file name in the first column
        try:
            image = Image.open(img_name).convert('RGB')  # Ensure the image is RGB
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None
        
        label = self.data_frame.iloc[idx, 1]  # Label in the second column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to tensor
])

# Function to create the CNN model
def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main execution block
if __name__ == '__main__':
    # Define paths using os.path.join
    base_dir = 'Data'
    csv_path = os.path.join(base_dir, 'APTOS-2019 Dataset', 'binary_train.csv')
    image_dir = os.path.join(base_dir, 'APTOS-2019 Dataset', 'train_images')

    # Load the dataset using the custom Dataset class
    print("Loading dataset...")
    dataset = FundusDataset(csv_file=csv_path, img_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Extract images and labels
    try:
        images, labels = next(iter(dataloader))
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)  # Exit if data loading fails

    # Convert tensors to numpy arrays for compatibility with Keras
    images = images.numpy().transpose((0, 2, 3, 1))  # Convert to HWC format (Height, Width, Channels)
    labels = np.array(labels)

    # Check if images and labels have been loaded correctly
    print(f"Loaded {len(images)} images and {len(labels)} labels.")

    # Cross validation and training
    input_shape = (224, 224, 3)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_curves = []
    accuracies = []

    # Plotting setup
    plt.figure(figsize=(10, 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"Training on fold {fold + 1}...")
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = create_model(input_shape)

        # Callbacks
        checkpoint = ModelCheckpoint(f'best_model_fold_{fold + 1}.keras', save_best_only=True, monitor='val_loss', mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)


        # Train the model and display progress
        print(f"Starting training for fold {fold + 1}...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, callbacks=[checkpoint, early_stop], verbose=1) 

        # Plot accuracy over time
        plt.plot(history.history['accuracy'], label=f'Fold {fold + 1} Accuracy')

        # ROC curve
        y_val_pred = model.predict(X_val).ravel()
        fpr, tpr, _ = roc_curve(y_val, y_val_pred)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr, roc_auc))
        accuracies.append(max(history.history['val_accuracy']))

    # Save the final model
    model.save('final_trained_model.keras')
    print("Final model saved.")

    # Plot ROC curves
    plt.figure(figsize=(10, 5))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        plt.plot(fpr, tpr, label=f'Fold {i + 1} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # Plot accuracy changes
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
