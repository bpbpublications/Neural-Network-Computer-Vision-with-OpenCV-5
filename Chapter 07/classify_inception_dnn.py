import os
import numpy as np
import cv2
# File paths to the InceptionV3 model weights and class names
imagenet_classes_filepath = "../weights/7/ILSVRC2012.txt"
inceptionv3_weights_filepath = "../weights/7/inceptionv3/inceptionv3.pb"

# Shape of the input image expected by the InceptionV3 model
inceptionv3_shape = (299, 299)

# Function to decode and format the predicted labels
def decode_predictions(predictions, class_names, top=5):
    results = []
    top_indices = predictions[0].argsort()[-top:][::-1]
    for i in top_indices:
        result = class_names[i] + ": " + str(predictions[0][i])
        results.append(result)
    return results

# Function to classify an image using OpenCV's dnn module and InceptionV3
def classify_image_using_opencvdnn(imagepath):
    # Read the class names from the provided file
    imagenet_class_names = None
    with open(imagenet_classes_filepath, 'rt') as f:
        imagenet_class_names = f.read().rstrip('\n').split('\n')

    # Load the InceptionV3 model from disk
    model = cv2.dnn.readNet(inceptionv3_weights_filepath)

    # Read and preprocess the input image
    im = cv2.imread(imagepath)
    resized_image = cv2.resize(im, inceptionv3_shape)
    image_blob = cv2.dnn.blobFromImage(resized_image, 1/127.5, inceptionv3_shape, [127.5, 127.5, 127.5])

    # Set the input blob for the model and perform forward pass
    model.setInput(image_blob)
    predictions = model.forward()

    # Return the decoded predictions using the decode_predictions function
    return decode_predictions(predictions, imagenet_class_names, 7)

# Main script execution
if __name__ == "__main__":
    # Call the classify_image_using_opencvdnn function with the specified image file path
    labels_and_confidences = classify_image_using_opencvdnn("../input_images/aeroplane.jpg")

    # Iterate over the list of labels and confidences and print each label with confidence
    for label in labels_and_confidences:
        print(label)
