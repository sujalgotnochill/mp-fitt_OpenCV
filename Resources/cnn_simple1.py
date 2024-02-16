import numpy as np
import tensorflow as tf
import mnist
from tensorflow.keras.models import Sequential
#from ann_visualizer.visualize import ann_viz
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from keras.callbacks import CSVLogger

csv_logger = CSVLogger("model_history_log.csv", append=True)

# model = model()
# ann_viz(model, title="")


# def model():
#   model = keras.models.Sequential()

#loding tyhe images  
train_images = mnist.train_images()
train_labels = mnist.train_labels()
print(len(train_labels))
test_images = mnist.test_images()
test_labels = mnist.test_labels()
print(len(test_labels))

# Normalize the images. Pixel values are between 0-255 in image learning it is 
# good practice to normalize your data to a smaller range like between 0 and 1.
train_images = (train_images / 255) - 0.5
print(train_images)
test_images = (test_images / 255) - 0.5
print(test_images)

#lets give the hyper parameters
num_filters = 1   
filter_size = 9
pool_size = 2

# Model is being trained on 1875 batches of 32 images each, not 1875 images. 1875*32 = 60000 images
# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)



tf.keras.callbacks.EarlyStopping(
      monitor="loss",
      min_delta=0,
      patience=0,
      verbose=0,
      mode="auto",
      baseline=None,
      restore_best_weights=False,
    )
    
callback_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)


# Train the model.
history = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  validation_data=(test_images, to_categorical(test_labels)),
  callbacks=[csv_logger, callback_1],
)

model.summary()

#Save the model:
model.save_weights('cnn.h5')


# Load the model's saved weights.
model.load_weights('cnn.h5')


# Using the trained model to make predictions is easy: we pass an array of inputs to predict() and it returns an array of outputs. Keep in mind that the output of our network is 10 probabilities (because of softmax), so we’ll use np.argmax() to turn those into actual digits.

# Predict on the first 5 test images.
predictions = model.predict(test_images[:10])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:10]) # [7, 2, 1, 0, 4]


#Callback records events into a History object.

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#The range of epochs, from 1 to the total number of epochs, is often used to plot the learning curves of the model.
#the range function is used to generate a list of integers from 1 to the length of the accuracy list plus one. 
# This is because the accuracy list is typically recorded at the end of each epoch, starting from the first epoch.
epochs = range(1, len(accuracy) + 1)



plt.plot(epochs, accuracy, 'bo', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()