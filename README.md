# Digit_Recognition_Using_CNN
* This project uses a simple Convolutional Neural Network (CNN) implemented using TensorFlow/Keras to classify handwritten digits from theMNIst Dataset.
* The model is trained and saved, and a separate script is provided to make predictions on custom images.

digit_recognition/
│
├── Train_Model.py # code to train and save the CNN model.
├── Predict_Digit.py # code to load the model and predict digits from images.
├── CNN_model.h5 # Saved model (generated after training).
├── 9.png # Sample input image (28x28 grayscale digit image).
├── README.md # Project documentation.

# Requirements

Install the required Python libraries using pip:

'pip3 install tensorflow keras matplotlib numpy'