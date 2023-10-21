import cv2
import numpy as np

def process_detection(image, detection):
    # YOLO returns values between 0 and 1. This value has to be scaled to suit the image size.
    # This is the reverse of standardizing the data.
    height, width, channels = image.shape
    center_x = int(detection[0] * width)
    center_y = int(detection[1] * height)
    object_width = int(detection[2] * width)
    object_height = int(detection[3] * height)

    # Rectangle coordinates
    topleft_x = int(center_x - object_width/2)
    topleft_y = int(center_y - object_height/2)

    return (topleft_x, topleft_y, object_width, object_height)



def detect_coco80objects_using_opencvdnn(impath, confidence_threshold = 0.5):
    
    scaling_factor = 1/255
    yolo_shape = (416,416)
    
    # Load Yolo
    model = cv2.dnn.readNet("../weights/8/yolo/yolov3.weights", "../weights/8/yolo/yolov3.cfg")

    # Read the COCO class names
    with open("../weights/8/yolo/coco.names", 'r') as file:
        lines = file.readlines()
        classes = [line.strip() for line in lines]    

    image = cv2.imread(impath)
    blob = cv2.dnn.blobFromImage(image, scaling_factor, yolo_shape, (0, 0, 0))


    # get all the layer names of the model
    layer_names = model.getLayerNames()
    # filter and choose only the output layers
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    # Detecting objects
    model.setInput(blob)
    results = model.forward(output_layers)

    # results is a tuple. It's length is equal to the number of output layers in the model.
    object_classes = []
    object_confidences = []
    object_coordinates = []

    
    for one_layer in results:
        # each layer can have multiple detections. Cycle through them all.
        for one_detection in one_layer:
            # Each detection is a  1d array. The contents are as explained below
            # 1st position (index 0 for Python) - x-coordinate of the bounding box's centroid of the detected object
            # 2nd position (index 1 for Python) - y-coordinate of the bounding box's centroid of the detected object
            # 3rd position                      - width of the bounding box
            # 4th position                      - height of the bounding box
            # 5th till end                      - Confidence level for each class of detected object. In case of this program,
            #                                     it is the COCO80 dataset
            confidence_scores_for_classes = one_detection[5:]
            classid_with_highest_confidence = np.argmax(confidence_scores_for_classes)
            class_confidence = confidence_scores_for_classes[classid_with_highest_confidence]

            if class_confidence > confidence_threshold:
                object_location = process_detection(image, one_detection)
                object_coordinates.append(object_location)
                object_confidences.append(float(class_confidence))
                object_classes.append(classes[classid_with_highest_confidence])


    # Now we are left with only objects that meet the confidence threshold. However, there can still be multiple detections 
    # for the same object with overlapping areas.
    # So, we need to de-duplicate them. We shall do so using the Non Maximum Suppression algorithm.
    indexes = cv2.dnn.NMSBoxes(object_coordinates, object_confidences, 0.4, 0.3)


    # indexes contains the objects which are of interest to us. Cycle through the indexes and calculate the coordinates in a way
    # that OpenCV can understand them.
    objects_and_locations = []
    for inx in indexes:
        class_label = object_classes[inx]
        (x,y,width,height) = object_coordinates[inx]
        top_left_coordinate = (x, y)
        bottom_right_coordinate = (x + width, y + height)

        one_object = {}
        one_object["class"] = class_label
        one_object["top_left"] = top_left_coordinate
        one_object["bottom_right"] = bottom_right_coordinate
        objects_and_locations.append(one_object)


    return objects_and_locations



def draw_outlines_around_detections(impath, objects_and_locations):
    image = cv2.imread(impath)
    for one_object in objects_and_locations:
        cv2.rectangle(image, one_object["top_left"], one_object["bottom_right"], (255,255,255), 3)
        cv2.putText(image, one_object["class"], one_object["top_left"], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    return image

# Main script execution
if __name__ == "__main__":
    image_path = "../input_images/test_image1.jpeg"
    
    objects_and_locations = detect_coco80objects_using_opencvdnn(image_path)

    image = draw_outlines_around_detections(image_path, objects_and_locations)
    cv2.namedWindow("object detections", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("object detections", image)
    cv2.waitKey(0)


