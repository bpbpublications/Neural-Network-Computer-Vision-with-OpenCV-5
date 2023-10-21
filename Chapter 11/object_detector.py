import cv2
import time
import numpy as np

class ObjectDetector:
    def __init__(self, object_detection_model_file, class_labels_file, confidence_threshold):
        """
        Initialize the ObjectDetector object.

        Args:
            object_detection_model_file (str): File path to the object detection model.
            class_labels_file (str): File path to the class labels file.
            confidence_threshold (float): Confidence threshold for object detection.

        Raises:
            FileNotFoundError: If the provided file paths do not exist.
        """
        # Check if the provided files exist
        if not all(map(lambda f: cv2.os.path.exists(f), [object_detection_model_file, class_labels_file])):
            raise FileNotFoundError("One or more provided file paths do not exist for object detection model.")

        # Initialize member objects. This should happen once per program execution to avoid repeated disk reads
        self.__model = cv2.dnn.readNet(object_detection_model_file)
        self.__classes = self.__load_labels(class_labels_file)
        self.__confidence_threshold = confidence_threshold

        # Get all the layer names of the model
        self.__layer_names = self.__model.getLayerNames()
        # Filter and choose only the output layers
        self.__output_layers = [self.__layer_names[i - 1] for i in self.__model.getUnconnectedOutLayers()]

    def __load_labels(self, labels_file):
        try:
            # Check if the labels file exists
            if not cv2.os.path.exists(labels_file):
                raise FileNotFoundError(f"Labels file '{labels_file}' does not exist.")

            # Read the COCO class names
            with open(labels_file, 'r') as file:
                lines = file.readlines()
                classes = [line.strip() for line in lines]

            return classes

        except Exception as e:
            print(f"Error in load_labels: {str(e)}")

    def __process_detection(self, image, yolo_shape, detection):
        height, width, channels = image.shape
        x_scaling = width / yolo_shape[0]
        y_scaling = height / yolo_shape[1]

        center_x = detection[0]
        center_y = detection[1]
        object_width = detection[2]
        object_height = detection[3]

        # Rectangle coordinates
        topleft_x = int(x_scaling * (center_x - object_width / 2))
        topleft_y = int(y_scaling * (center_y - object_height / 2))

        object_width = int(x_scaling * object_width)
        object_height = int(y_scaling * object_height)

        return (topleft_x, topleft_y, object_width, object_height)

    def detect_objects(self, original_image):
        """
        Detect objects in an input image.

        Args:
            original_image (numpy.ndarray): An OpenCV image object.

        Returns:
            numpy.ndarray: An image with objects marked.

        Raises:
            ValueError: If the provided image is not a valid numpy.ndarray.
        """
        scaling_factor = 1 / 255
        nms_threshold = 0.1
        yolo_shape = (640, 640)

        try:
            # Ensure the image is a valid numpy.ndarray
            if not isinstance(original_image, np.ndarray):
                raise ValueError("Input image is not a valid numpy.ndarray.")

            image = np.copy(original_image)

            # Perform object detection
            blob = cv2.dnn.blobFromImage(image, scaling_factor, yolo_shape, (0, 0, 0), 1, crop=False)

            # Detecting objects
            self.__model.setInput(blob)
            results = self.__model.forward(self.__output_layers)

            # Initialize lists to store object information
            object_classes = []
            object_confidences = []
            object_coordinates = []

            number_of_detections = results[0].shape[1]
            for inx in range(number_of_detections):
                one_detection = results[0][0][inx]

                # Each detection is a 1D array with the following format:
                # [x, y, width, height, confidence_score_for_class_0, confidence_score_for_class_1, ...]
                # We extract the class with the highest confidence score.
                confidence_scores_for_classes = one_detection[5:]
                classid_with_highest_confidence = np.argmax(confidence_scores_for_classes)
                class_confidence = confidence_scores_for_classes[classid_with_highest_confidence]

                if class_confidence > self.__confidence_threshold:
                    object_location = self.__process_detection(image, yolo_shape, one_detection)
                    object_coordinates.append(object_location)
                    object_confidences.append(float(class_confidence))
                    object_classes.append(self.__classes[classid_with_highest_confidence])

            # Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
            indexes = cv2.dnn.NMSBoxes(object_coordinates, object_confidences, self.__confidence_threshold, nms_threshold)

            # Prepare the final list of objects and their coordinates
            objects_and_locations = []
            for inx in indexes:
                class_label = object_classes[inx]
                (x, y, width, height) = object_coordinates[inx]
                top_left_coordinate = (x, y)
                bottom_right_coordinate = (x + width, y + height)

                one_object = {}
                one_object["class"] = class_label
                one_object["top_left"] = top_left_coordinate
                one_object["bottom_right"] = bottom_right_coordinate
                one_object["confidence"] = object_confidences[inx]
                objects_and_locations.append(one_object)

            # Draw bounding boxes and class labels on the original image
            for one_object in objects_and_locations:
                cv2.rectangle(original_image, one_object["top_left"], one_object["bottom_right"], (255, 255, 255), 3)
                cv2.putText(original_image, one_object["class"], one_object["top_left"], cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)

            result_image = original_image

        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            result_image = None

        return result_image
