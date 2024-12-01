# data loading and preprocessing goes here
# loading the data set
# augmenting and preprocessing images
# splitting the data set into training and test sets

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# avoiding out of memory errors
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists('data/train'):
    raise FileNotFoundError("The directory 'data/train' does not exist. Please check the path.")
data = tf.keras.utils.image_dataset_from_directory('data/train')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

print(batch[0].max())

class_names = data.class_names
print("Class names:", class_names)

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    label = batch[1][idx]
    ax[idx].title.set_text(class_names[label])
plt.show()

# data preprocessing 
scaled_data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator() # with shuffling, so data is changing
batch = scaled_iterator.next()

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

train = scaled_data.take(train_size)
val = scaled_data.skip(train_size).take(val_size)
test = scaled_data.skip(train_size + val_size)

print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
