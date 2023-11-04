import keras
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import rmsprop_v2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Create empty lists to store the training and test images
train_images = []
train_labels = []
test_images = []
test_labels = []

# Loop through the training folders
for folder in ['with_mask', 'without_mask']:
    # List all the file names in the current training folder
    files = os.listdir('Dataset/train/' + folder)

    # Loop through the file names
    for file in files:
        # Load the image
        img = cv2.imread('Dataset/train/' + folder + '/' + file)

        # Resize the image
        img = cv2.resize(img, (100, 100))

        # Convert the image to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # # Normalize the pixel values
        # mean, std = cv2.meanStdDev(img)
        # img = (img - mean) / std


        # Add the image to the list of training images
        train_images.append(img)

        # Add the corresponding label to the list of training labels
        if folder == 'with_mask':
            train_labels.append(1)
        else:
            train_labels.append(0)

# Repeat the same process for the test set
for folder in ['with_mask', 'without_mask']:
    files = os.listdir('Dataset/test/' + folder)
    for file in files:
        img = cv2.imread('Dataset/test/' + folder + '/' + file)
        img = cv2.resize(img, (100, 100))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mean, std = cv2.meanStdDev(img)
        # img = (img - mean) / std
        test_images.append(img)
        if folder == 'with_mask':
            test_labels.append(1)
        else:
            test_labels.append(0)



#print some images
# indices = [0,1,2,3,4]
#
# # Loop through the indices
# for i in indices:
#     # Get the image and its label
#     img = train_images[i]
#     label = train_labels[i]
#
#     # Print the label
#     print(label)
#
#     # Display the image
#     plt.imshow(img,cmap="gray")
#     plt.show()


# Convert the images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Reshape the images to have an additional channel dimension
# train_images = train_images.reshape((train_images.shape[0], 100, 100, 1))
# test_images = test_images.reshape((test_images.shape[0], 100, 100, 1))


# Normalize the pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0


# Load the base model
base_model = tf.keras.applications.VGG16(input_shape=(100, 100, 3), include_top=False, weights='imagenet')


# Add a few layers on top of the base model
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# freeze the layers of VGG
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
#model.compile(optimizer=rmsprop_v2.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=5,
                    batch_size=16)


# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels)



# model=Sequential()
#
# model.add(Conv2D(100,(3,3),input_shape=train_images.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(100,(3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(1,activation='softmax'))




#
#
# # Define the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=1, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# checkpoint = ModelCheckpoint(
#   'model-{epoch:03d}.model',
#   monitor='val_loss',
#   verbose=0,
#   save_best_only=True,
#   mode='auto')
#
# history=model.fit(
#   train_images,
#   train_labels,
#   epochs=20,
#   callbacks=[checkpoint],
#   validation_split=0.2)
#
# print(model.evaluate(test_images,test_labels))
