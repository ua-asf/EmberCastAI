import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Configs
# -----------------------------
FOLDER_PATH = "organized_dataset/Caldor/"
IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 5
OUTPUT_PREDICTION = "predicted_perimeter_red_transparent.png"
EPOCHS = 20
BATCH_SIZE = 4
THRESHOLD = 0.5

# -----------------------------
# Loss: Combined Dice + Binary Crossentropy
# -----------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# -----------------------------
# Image Helpers
# -----------------------------
def red_mask_to_binary(img):
    img = img.convert("RGB")
    red_channel = np.array(img)[:, :, 0]
    binary_mask = (red_channel > 150).astype(np.uint8)
    return binary_mask

def binary_mask_to_red_transparent(mask_img):
    red = Image.new("RGBA", mask_img.size, (255, 0, 0, 0))
    mask_img = mask_img.convert("L")
    alpha = mask_img.point(lambda p: 255 if p > 128 else 0)
    r, g, b, a = red.split()
    return Image.merge("RGBA", (r, g, b, alpha))

# -----------------------------
# Data Loader for Full Mask Prediction
# -----------------------------
def load_fire_sequence_recursive(root_dir, img_size=(128, 128), sequence_length=5):
    png_files = []
    original_sizes = []

    for root, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.endswith(".png"):
                png_files.append(os.path.join(root, file))

    imgs = []
    for file_path in png_files:
        original_img = Image.open(file_path)
        original_sizes.append(original_img.size)

        mask = red_mask_to_binary(original_img)
        mask = Image.fromarray(mask).resize(img_size, Image.NEAREST)
        mask_array = np.array(mask).astype(np.float32) / 255.0
        imgs.append(mask_array)

    imgs = np.array(imgs)[..., np.newaxis]

    sequences = []
    targets = []
    for i in range(len(imgs) - sequence_length):
        sequences.append(imgs[i:i+sequence_length-1])  # Input [t0 to t3]
        targets.append(imgs[i+sequence_length-1])      # Label t4

    return np.array(sequences), np.array(targets), png_files, original_sizes[SEQUENCE_LENGTH:]

# -----------------------------
# ConvLSTM Model
# -----------------------------
def build_model(input_shape):
    model = Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss=combined_loss, metrics=["accuracy"])
    return model

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading data...")
    X, y, file_paths, original_sizes = load_fire_sequence_recursive(FOLDER_PATH, IMG_SIZE, SEQUENCE_LENGTH)
    if len(X) == 0:
        print("Not enough data.")
        return

    print(f"Training on {len(X)} sequences...")
    model = build_model(input_shape=X.shape[1:])
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stop])

    print("Predicting next mask...")
    predicted = model.predict(X[-1:])[0].squeeze()

    plt.imshow(predicted, cmap="hot")
    plt.title("Predicted Fire Perimeter (Raw Output)")
    plt.colorbar()
    plt.show()

    predicted_mask = (predicted > THRESHOLD).astype(np.uint8)
    predicted_mask_img = Image.fromarray(predicted_mask * 255)

    original_size = original_sizes[-1]
    predicted_resized = predicted_mask_img.resize(original_size, Image.NEAREST)
    red_transparent = binary_mask_to_red_transparent(predicted_resized)

    print(f"Saving prediction to {OUTPUT_PREDICTION}")
    red_transparent.save(OUTPUT_PREDICTION)
    print("Done!")

if __name__ == "__main__":
    main()