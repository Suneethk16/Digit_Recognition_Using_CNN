import tensorflow as tf 
import numpy as np
from keras.preprocessing import image 
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('CNN_model.h5')

img_path = '9.png'  #replace with your image
img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

print(f"Predicted digit: {predicted_digit}")
plt.imshow(img_array.reshape(28,28), cmap='gray')
plt.title(f'Predicted Digit: {predicted_digit}')
plt.axis('off')
plt.show()
