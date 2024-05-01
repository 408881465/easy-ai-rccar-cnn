import tensorflow as tf
# tf.__version__

# import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization
# tf.keras.layers.BatchNormalization
from  tensorflow.keras.preprocessing.image  import  ImageDataGenerator

# import zipfile
# with zipfile.ZipFile('/workspace/gitpod-tensorflowjs-to-arduino/rccardataset1209.zip', 'r') as zip_ref:
#     zip_ref.extractall('/workspace/gitpod-tensorflowjs-to-arduino/')


data_dir = '/workspace/easy-ai-rccar-cnn/self-drving-rccar-dataset'
batch_size = 32
img_height = 224
img_width = 224
BATCH_SIZE = batch_size
IMAGE_SIZE = (img_height,img_width)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
    

AUTOTUNE = tf.data.experimental.AUTOTUNE
# https://tf.wiki/zh_hans/basic/tools.html
# tf.data 的数据集对象为我们提供了 Dataset.prefetch() 方法，使得我们可以让数据集对象 Dataset 在训练时预取出若干个元素，使得在 GPU 训练的同时 CPU 可以准备数据，从而提升训练流程的效率。
print(AUTOTUNE)
# 此处参数 buffer_size 既可手工设置，也可设置为 tf.data.experimental.AUTOTUNE 从而由 TensorFlow 自动选择合适的数值。
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# num_classes = 3

model6 = Sequential()
model6.add(BatchNormalization(input_shape=(img_height, img_width, 3)))
model6.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model6.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model6.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))

# model.add(Dropout(0.5))
model6.add(Conv2D(64, (3, 3), activation='elu'))
# model.add(Dropout(0.3))
model6.add(Conv2D(64, (3, 3), activation='elu'))
model6.add(Dropout(0.5))
# model.add(Dropout(0.3))
model6.add(Conv2D(128, (3, 3), activation='elu'))
model6.add(Dropout(0.5))
model6.add(Flatten())
model6.add(Dense(125, activation='elu'))
model6.add(Dropout(0.25))
model6.add(Dense(60, activation='elu'))
model6.add(Dropout(0.25))
model6.add(Dense(25, activation='elu'))
model6.add(Dropout(0.25))
model6.add(Dense(3))

model6.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model6.summary()

epochs=10
history3 = model6.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model6.save('rccar6.h5')