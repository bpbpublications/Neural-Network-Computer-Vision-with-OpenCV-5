import cv2

def  detect_coco80objects_using_opencvdnn(image_path, confidence_threshold):
    ssd_size=(300, 300)

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Load SSD300
    net = cv2.dnn.readNetFromTensorflow("../weights/8/ssd300/ssd_mobilenet_v2_coco_2018_03_29.pb", "../weights/8/ssd300/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

    # Read the COCO class names
    with open("../weights/8/ssd300/coco.names", 'r') as file:
        lines = file.readlines()
        classes = [line.strip() for line in lines]  


    blob = cv2.dnn.blobFromImage(image, size=ssd_size, swapRB=True, crop=False)
    net.setInput(blob)
    results = net.forward()
    # print(results.shape)


    objects_and_locations = []
    for one_detection in results[0,0,:,:]:
        # Each detection is a  1d array. The contents are as explained below
        # 2nd position (index 1 for Python) - classid of the object. In case of this program, it is the COCO80 dataset
        # 3rd position (index 2 for Python) - Confidence level for detected class. 
        # 4th position        3             - Left most point of the bounding box
        # 5th position        4             - Top most point of the bounding box
        # 6th position        5             - Right most point of the bounding box
        # 7th position        6             - Bottom most point of the bounding box                                  

        confidence_score = float(one_detection[2])    
        if confidence_score <= confidence_threshold:
            continue

        left = int(one_detection[3] * width)
        top = int(one_detection[4] * height)
        right = int(one_detection[5] * width)
        bottom = int(one_detection[6] * height)

        class_ids = int(one_detection[1])
        class_label = classes[class_ids]
        one_object = {}
        one_object["class"] = class_label
        one_object["top_left"] = (left, top)
        one_object["bottom_right"] = (right, bottom)
        one_object["confidence"] = confidence_score
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
    image_path = "../input_images/aeroplane.jpg"
    confidence_threshold=0.5

    objects_and_locations = detect_coco80objects_using_opencvdnn(image_path, confidence_threshold)


    image = draw_outlines_around_detections(image_path, objects_and_locations)
    cv2.namedWindow("object detections", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("object detections", image)
    cv2.waitKey(0)



