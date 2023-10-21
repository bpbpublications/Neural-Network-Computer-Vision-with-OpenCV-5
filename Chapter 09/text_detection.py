import numpy as np
import cv2


def visualize(image, results):
    box_color=(255, 255, 255)
    isClosed=True
    thickness=10 
    pts = np.array(results[0])
    return cv2.polylines(image, pts, isClosed, box_color, thickness)


def initialize_model(model_shape):
    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = cv2.dnn.DNN_TARGET_CPU
    
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

if __name__ == '__main__':

    image_path = "../input_images/4_EdgesCorners.jpg"
    model_shape = (736, 736)  # w, h

    model = initialize_model(model_shape)

    original_image = cv2.imread(image_path)
    original_h, original_w, _  = original_image.shape
    scaleHeight = original_h / model_shape[1]
    scaleWidth = original_w / model_shape[0]
    image = cv2.resize(original_image, model_shape)

    # Inference
    results = model.detect(image)

    # Scale the results bounding box
    for i in range(len(results[0])):
        for j in range(4):
            box = results[0][i][j]
            results[0][i][j][0] = box[0] * scaleWidth
            results[0][i][j][1] = box[1] * scaleHeight

    # Draw results on the input image
    original_image = visualize(original_image, results)

    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.imshow("input", original_image)
    cv2.waitKey(0)
    cv2.imwrite("../output_images/text_detection.jpg", original_image)
