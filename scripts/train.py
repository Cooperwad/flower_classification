# training and evaluation
# saving results into results folder
# pulls from data_prep for data and model for our architecture

import tensorflow as tf
import pickle
import os
import data_prep
from constants import MODEL_PATH, HISTORY_PATH
import model
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


model = model.model
train = data_prep.train
val = data_prep.val

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
    plt.plot(history_dict['accuracy'], color='blue', label='Accuracy')
    plt.plot(history_dict['val_accuracy'], color='red', label='Validation Accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
else:
    print("No history available to plot.")

