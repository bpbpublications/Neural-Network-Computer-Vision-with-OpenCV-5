# Import necessary libraries
import os
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image as image_utils
from keras import utils

# Function to preprocess an image for MobileNetV2
def preprocess_image(im):
    img = utils.load_img(im, target_size=(224, 224))
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to classify an image using MobileNetV2
def classify_image_using_tensorflow(imagepath):
    predicted_labels = []

    # Load the pre-trained MobileNetV2 model with weights from the ImageNet dataset
    model = MobileNetV2(include_top=True, weights='imagenet')

    # Preprocess the input image using the preprocess_image function
    preprocessed_image = preprocess_image(imagepath)

    # Make a prediction using the pre-trained model on the preprocessed image
    pred = model.predict(preprocessed_image)

    # Decode and process the prediction to get the top predicted labels
    for prediction in decode_predictions(pred, top=7)[0]:
        predicted_labels.append(prediction)

    # Return the list of predicted labels
    return predicted_labels

# Main script execution
if __name__ == "__main__":
    # Call the classify_image_using_tensorflow function with the specified image file path
    labels = classify_image_using_tensorflow("../input_images/aeroplane.jpg")

    # Iterate over the list of predicted labels and print each label
    for l in labels:
        print(l)
