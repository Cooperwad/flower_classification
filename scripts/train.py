# training and evaluation
# saving results into results folder
# pulls from data_prep for data and model for our architecture

import tensorflow as tf
import pickle
import os
import data_prep
import model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import load_model


model = model.model
train = data_prep.train
val = data_prep.val

MODEL_PATH = 'flower_model_2.h5'
HISTORY_PATH = 'history_2.pkl'


logdir='logs'
callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Loaded saved model!")
    # if history.pkl exists, load the training history
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        print(f"Training history loaded from {HISTORY_PATH}")
    else:
        history = None
        print("No training history found.")
else:
    # train the model and save it
    history = model.fit(train, epochs=20, validation_data=val, callbacks=[callback])
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {HISTORY_PATH}")

# plot training history (if available)
if history:
    if isinstance(history, dict):
        history_dict = history
    else:
        history_dict = history.history

    fig = plt.figure()
    plt.plot(history_dict['loss'], color='blue', label='Training Loss')
    plt.plot(history_dict['val_loss'], color='red', label='Validation Loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
else:
    print("No history available to plot.")

