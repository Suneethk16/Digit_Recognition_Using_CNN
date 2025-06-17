"""
This code predicts the digit that is provided through the input image using 
Convolutional Neural Network(CNN) model and this model has been trained using the MNIST dataset.
It displays the image along with the predicted digit.

Steps:
1. Load the pre-trained CNN model.
2. Load and preprocess the input image.
3. Use the model to predict the digit.
4. Print and display the prediction.
"""

import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

#Initialize the trained CNN model
model = tf.keras.models.load_model('CNN_model.h5')

#Change the path according to the digit you want to predict
img_path = 'Input_Data/9.png'
img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))

#Convert the image to a normalized array of NumPy
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

#Model uses the picture to predict the number
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

#Display the final result.
print(f"Predicted digit: {predicted_digit}")
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f'Predicted Digit: {predicted_digit}')
plt.axis('off')
plt.show()
