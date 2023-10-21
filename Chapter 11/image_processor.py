import cv2
import object_detector as detector
import numberplate_recognizor as numplaterecog

class ImageProcessor:
    def __init__(self, object_detection_model_file, labels_file,
                 textdetection_model_file, textrecognition_model_file,
                 confidence_threshold):
        """
        Initialize the ImageProcessor object.

        Args:
            object_detection_model_file (str): File path to the object detection model.
            labels_file (str): File path to the labels file.
            textdetection_model_file (str): File path to the text detection model.
            textrecognition_model_file (str): File path to the text recognition model.
            confidence_threshold (float): Confidence threshold for object detection.

        Raises:
            FileNotFoundError: If any of the provided file paths do not exist.
        """
        # Initialize member objects, e.g., load models or configure settings

        # Create an ObjectDetector instance for object detection
        self.__object_detection_model = detector.ObjectDetector(
            object_detection_model_file=object_detection_model_file,
            class_labels_file=labels_file,
            confidence_threshold=confidence_threshold
        )

        # Create a NumberPlateRecognizor instance for number plate recognition
        self.__numberplate_detection_model = numplaterecog.NumberPlateRecognizor(
            textdetection_model_file, textrecognition_model_file
        )

    def detect_objects(self, image):
        """
        Detect objects in an input image.

        Args:
            image (numpy.ndarray): An OpenCV image object.

        Returns:
            numpy.ndarray: An image with objects marked.

        Raises:
            Exception: If an error occurs during object detection.
        """
        try:
            # Perform object detection using self.__object_detection_model
            retimage = self.__object_detection_model.detect_objects(image)

        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            retimage = image  # Return the original image in case of an error

        return retimage

    def detect_numberplate(self, image):
        """
        Detect number plates in an input image.

        Args:
            image (numpy.ndarray): An OpenCV image object.

        Returns:
            str: Recognized number plate text.

        Raises:
            Exception: If an error occurs during number plate detection.
        """
        try:
            # Perform number plate detection using self.__numberplate_detection_model
            return self.__numberplate_detection_model.detect_numberplate(image)

        except Exception as e:
            print(f"Error in detect_numberplate: {str(e)}")
            return None  # Return None in case of an error
