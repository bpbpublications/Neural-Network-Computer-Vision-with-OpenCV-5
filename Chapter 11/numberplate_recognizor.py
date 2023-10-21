import cv2
import numpy as np

class NumberPlateRecognizor:

    # Constants for the model and image processing
    __MODEL_SHAPE = (736, 736)  # Model input shape (width, height)
    __BACKEND_ID = cv2.dnn.DNN_BACKEND_OPENCV
    __TARGET_ID = cv2.dnn.DNN_TARGET_CPU

    def __init__(self, textdetection_model_file, textrecognition_model_file):
        """
        Initialize the NumberPlateRecognizor object.

        Args:
            textdetection_model_file (str): File path to the text detection model.
            textrecognition_model_file (str): File path to the text recognition model.

        Raises:
            FileNotFoundError: If any of the provided file paths do not exist.
        """

        # Initialize member objects, e.g., load models or configure settings

        # Initialize the text detection model
        self.__detector_model = self.__initialize_textdetector_model(textdetection_model_file)

        # Initialize CRNN for text recognition
        self.__recognizer, self.__character_set, self.__character_size, self.__vertex_coordinates = self.__initialize_english_textrecognition_model(textrecognition_model_file)

    def __initialize_textdetector_model(self, model_path):
        # Constants for text detection parameters
        binary_threshold = 0.3
        polygon_threshold = 0.5
        max_candidates = 200
        unclip_ratio = 2.0

        # Create a text detection model
        model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(model_path))

        model.setPreferableBackend(self.__BACKEND_ID)
        model.setPreferableTarget(self.__TARGET_ID)

        model.setBinaryThreshold(binary_threshold)
        model.setPolygonThreshold(polygon_threshold)
        model.setUnclipRatio(unclip_ratio)
        model.setMaxCandidates(max_candidates)

        model.setInputParams(1.0/255.0, self.__MODEL_SHAPE, (122.67891434, 116.66876762, 104.00698793))
        return model

    def __initialize_english_textrecognition_model(self, model_path):
        # Create a text recognition model
        model = cv2.dnn.readNet(model_path)
        model.setPreferableBackend(self.__BACKEND_ID)
        model.setPreferableTarget(self.__TARGET_ID)

        # Define character set and size
        character_set = '0123456789abcdefghijklmnopqrstuvwxyz'
        character_size = (100, 32)  # This must not be changed and must be in sync with next line
        vertex_coordinates = np.array([
            [0, 31],
            [0, 0],
            [99, 0],
            [99, 31]
        ],
        dtype=np.float32)

        return model, character_set, character_size, vertex_coordinates

    def __recognize_text(self, image, boxshape):
        # Preprocess the image
        vertices = boxshape.reshape((4, 2)).astype(np.float32)
        rotationMatrix = cv2.getPerspectiveTransform(vertices, self.__vertex_coordinates)
        cropped_image = cv2.warpPerspective(image, rotationMatrix, self.__character_size)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        text_blob = cv2.dnn.blobFromImage(cropped_image, size=self.__character_size, mean=127.5, scalefactor=1 / 127.5)

        # Forward pass
        self.__recognizer.setInput(text_blob)
        output_blob = self.__recognizer.forward()

        # Postprocess the recognized text
        text = ''
        for i in range(output_blob.shape[0]):
            c = np.argmax(output_blob[i][0])
            if c != 0:
                text += self.__character_set[c - 1]
            else:
                text += '-'

        # Return processed text
        char_list = []
        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])

        return ''.join(char_list)

    def __visualize(self, image, boxes, texts):
        # Visualize the recognized text on the image
        color = (255, 255, 255)
        isClosed = True
        thickness = 2
        pts = np.array(boxes[0])
        output = cv2.polylines(image, pts, isClosed, color, thickness)
        for box, text in zip(boxes[0], texts):
            cv2.putText(output, text, (box[1].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        return output

    def detect_numberplate(self, original_image):
        """
        Detect number plates in an input image.

        Args:
            original_image (numpy.ndarray): An OpenCV image object.

        Returns:
            numpy.ndarray: An image with number plates marked.

        Raises:
            ValueError: If the provided image is not a valid numpy.ndarray.
        """
        try:
            # Ensure the image is a valid numpy.ndarray
            if not isinstance(original_image, np.ndarray):
                raise ValueError("Input image is not a valid numpy.ndarray.")

            # Get the original image dimensions
            original_h, original_w, _ = original_image.shape
            scaleHeight = original_h / self.__MODEL_SHAPE[1]
            scaleWidth = original_w / self.__MODEL_SHAPE[0]

            # Resize the image to the model's input shape
            image = cv2.resize(original_image, self.__MODEL_SHAPE)

            # Detect the locations of text in the resized image
            results = self.__detector_model.detect(image)

            # Recognize text in the detected locations
            texts = []
            for box, score in zip(results[0], results[1]):
                text = self.__recognize_text(image, box.reshape(8))
                texts.append(text)

            # Scale the results bounding box back to the original image dimensions
            for i in range(len(results[0])):
                for j in range(4):
                    box = results[0][i][j]
                    results[0][i][j][0] = box[0] * scaleWidth
                    results[0][i][j][1] = box[1] * scaleHeight

            # Draw results on the original input image
            original_image = self.__visualize(original_image, results, texts)
            return original_image

        except Exception as e:
            print(f"Error in detect_numberplate: {str(e)}")
