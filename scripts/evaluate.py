import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import data_prep
from constants import MODEL_PATH, HISTORY_PATH

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found.")
    exit()

model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

test = data_prep.test
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # plot Loss
    ax[0].plot(history['loss'], label='Training Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss Over Epochs')
    ax[0].legend()

    # plot Accuracy
    if 'accuracy' in history:
        ax[1].plot(history['accuracy'], label='Training Accuracy')
        ax[1].plot(history['val_accuracy'], label='Validation Accuracy')
        ax[1].set_title('Accuracy Over Epochs')
        ax[1].legend()

    plt.show()
else:
    print(f"Training history file {HISTORY_PATH} not found.")
