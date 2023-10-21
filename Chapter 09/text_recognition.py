import numpy as np
import cv2

backend_id = cv2.dnn.DNN_BACKEND_OPENCV
target_id = cv2.dnn.DNN_TARGET_CPU


def initialize_textdetector_model(model_shape):
    binary_threshold = 0.3
    polygon_threshold = 0.5
    max_candidates = 200
    unclip_ratio = 2.0
     
    model_path = "../weights/9/text_detection_DB_TD500_resnet18_2021sep.onnx"
    model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(model_path))

    model.setPreferableBackend(backend_id)
    model.setPreferableTarget(target_id)

    model.setBinaryThreshold(binary_threshold)
    model.setPolygonThreshold(polygon_threshold)
    model.setUnclipRatio(unclip_ratio)
    model.setMaxCandidates(max_candidates)

    model.setInputParams(1.0/255.0, model_shape, (122.67891434, 116.66876762, 104.00698793))
    return model

def initialize_english_textrecognition_model():
    model_path = "../weights/9/text_recognition_CRNN_EN_2021sep.onnx"
    model = cv2.dnn.readNet(model_path)
    model.setPreferableBackend(backend_id)
    model.setPreferableTarget(target_id)
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

def visualize(image, boxes, texts, color=(0, 255, 0), isClosed=True, thickness=2):
    pts = np.array(boxes[0])
    output = cv2.polylines(image, pts, isClosed, color, thickness)
    for box, text in zip(boxes[0], texts):
        print(text)
        cv2.putText(output, text, (box[1].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    return output

def recognize_text(model, character_set, character_size, vertex_coordinates, image, boxshape):
    # Preprocess the image
    
    # Remove conf, reshape and ensure all is np.float32
    vertices = boxshape.reshape((4, 2)).astype(np.float32)
    rotationMatrix = cv2.getPerspectiveTransform(vertices, vertex_coordinates)
    cropped_image = cv2.warpPerspective(image, rotationMatrix, character_size)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    text_blob = cv2.dnn.blobFromImage(cropped_image, size=character_size, mean=127.5, scalefactor=1 / 127.5)



    # Forward
    model.setInput(text_blob)
    output_blob = model.forward()

    # Postprocess
    text = ''
    for i in range(output_blob.shape[0]):
        c = np.argmax(output_blob[i][0])
        if c != 0:
            text += character_set[c - 1]
        else:
            text += '-'

    # return text
    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])

    return ''.join(char_list)

if __name__ == '__main__':
    image_path = "../input_images/4_EdgesCorners.jpg"
    model_shape = (736, 736)  # w, h

    # initialize text detection model
    detector = initialize_textdetector_model(model_shape)



    # initialize CRNN for text recognition
    recognizer, character_set, character_size, vertex_coordinates = initialize_english_textrecognition_model()

    original_image = cv2.imread(image_path)
    original_h, original_w, _  = original_image.shape
    scaleHeight = original_h / model_shape[1]
    scaleWidth = original_w / model_shape[0]
    image = cv2.resize(original_image, model_shape)

    # Detect the locations of text
    results = detector.detect(image)


    # Recognize text in the detected locations
    texts = []
    for box, score in zip(results[0], results[1]):
        text = recognize_text(recognizer, character_set, character_size, vertex_coordinates, image, box.reshape(8))
        texts.append(text)

    # Scale the results bounding box
    for i in range(len(results[0])):
        for j in range(4):
            box = results[0][i][j]
            results[0][i][j][0] = box[0] * scaleWidth
            results[0][i][j][1] = box[1] * scaleHeight

    # Draw results on the input image
    original_image = visualize(original_image, results, texts)

    # Visualize results in a new window
    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.imshow("input", original_image)
    cv2.waitKey(0)
    cv2.imwrite("../output_images/text_recognition.jpg", original_image)
