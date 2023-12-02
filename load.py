from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

model = keras.models.load_model("model.h5")

# Load the image and apply transformation to data 
img_array = cv2.imread('polyps.jpg')

print(img_array)

img_resize_rgb = cv2.resize(img_array, (100, 100))



# Add batch dimension
data = np.expand_dims(img_resize_rgb, axis=0)

# Make predictions
predictions = model.predict(data)

pred0 = predictions[0]

label0 = np.argmax(pred0)

print(label0)
