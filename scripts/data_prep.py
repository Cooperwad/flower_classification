# data loading and preprocessing goes here
# loading the data set
# augmenting and preprocessing images
# splitting the data set into training and test sets

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# avoiding out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

data = data.map(lambda x, y: (x/255, y))

