import numpy as np
from keras import utils 
from keras.applications.resnet import ResNet152, preprocess_input
from keras.applications.imagenet_utils import decode_predictions

def preprocess_image(im):
    # Load the image and resize it to the target size of (224, 224).
    img = utils.load_img(im, target_size=(224, 224))

    # Convert the loaded image to a NumPy array.
    img = utils.img_to_array(img)

    # Add an additional dimension to the array to represent batch size (1).
    img = np.expand_dims(img, axis=0)

    # Preprocess the image data for the specific deep learning model.
    img = preprocess_input(img)

    # Return the preprocessed image.
    return img



def classify_image_using_tensorflow(imagepath):
    # Create an empty list to store the predicted labels.
    predicted_labels = []

    # Load the pre-trained ResNet-152 model with weights from the ImageNet dataset.
    model = ResNet152(include_top=True, weights='imagenet')

    # Preprocess the input image using the preprocess_image function (not shown here).
    preprocessed_image = preprocess_image(imagepath)

    # Make a prediction using the pre-trained model on the preprocessed image.
    pred = model.predict(preprocessed_image)

    # Decode and process the prediction to get the top predicted labels.
    for prediction in decode_predictions(pred)[0]:
        predicted_labels.append(prediction)

    # Return the list of predicted labels.
    return predicted_labels

if __name__ == "__main__":
    # This block of code will be executed only if this script is run directly as the main program.

    # Call the classify_image_using_tensorflow function with the specified image file path.
    labels = classify_image_using_tensorflow("../input_images/aeroplane.jpg")

    # Iterate over the list of predicted labels and print each label.
    for l in labels:
        print(l)
