import pandas as pd
import os

train_csv_path = "Data/APTOS-2019 Dataset/train.csv"
binary_train_csv_path = "Data/APTOS-2019 Dataset/binary_train.csv" # Where the binary csv file is saved

df = pd.read_csv(train_csv_path)

df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1) # Converts diagnosis labels to binary (0 remains 0, 1-4 become 1)

df.to_csv(binary_train_csv_path, index=False)