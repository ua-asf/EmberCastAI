import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

# -----------------------------
# Configs
# -----------------------------

FOLDER_PATH = "dataset/png/calif_n/2021_FEDERAL_Incidents/CA-ENF-024030_Caldor"
IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 5
OUTPUT_FILE = "predicted_perimeter.png"

# -----------------------------
# Data Loader
# -----------------------------

def load_fire_sequence_recursive(root_dir, img_size=(128, 128), sequence_length=5):
    png_files = []

    # Walk through all subfolders
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                png_files.append(full_path)

    # Sort by path 
    png_files.sort()

    imgs = []
    for file_path in png_files:
        img = Image.open(file_path).convert("L")
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0  
        imgs.append(img_array)

    imgs = np.array(imgs)  
    imgs = np.expand_dims(imgs, axis=-1)  
    sequences = []
    for i in range(len(imgs) - sequence_length):
        seq = imgs[i:i+sequence_length]
        sequences.append(seq)

    return np.array(sequences)  

# -----------------------------
# ConvLSTM Model
# -----------------------------

def build_model(input_shape):
    model = Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')  # <-- FIXED
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


# -----------------------------
# Main
# -----------------------------

def main():
    print("Loading data")
    sequence_data = load_fire_sequence_recursive(FOLDER_PATH, IMG_SIZE, SEQUENCE_LENGTH)
    if len(sequence_data) == 0:
        print("Not enough images to create sequences.")
        return

    X = sequence_data[:, :-1]  
    y = sequence_data[:, -1]   

    print("Building model")
    model = build_model(input_shape=X.shape[1:])  

    print("Predicting next frame (random weights)")
    predicted = model.predict(X[-1:])[0]  

    print("Saving predicted perimeter image")
    predicted_mask = (predicted.squeeze() > 0.5).astype(np.uint8) * 255
    Image.fromarray(predicted_mask).save(OUTPUT_FILE)

    print(f"Saved prediction to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
