import numpy as np
import cv2

def visualize(input, faces):
    if faces is None:
        return
    
    if faces[1] is None:
        return

    thickness = 5
    facebox_color = (255, 255, 255)
    eyeline_color = (255, 0, 0)
    nosetip_color = (0, 255, 0)
    mouthline_color = (0, 0, 255)

    # Cycle through the faces and landmarks
    for _, face in enumerate(faces[1]):
        # Convert the returned coordinates to integers
        coordinates = face[:-1].astype(np.int32)

        face_box_topleft_x = coordinates[0]
        face_box_topleft_y = coordinates[1]
        face_box_width = coordinates[2]
        foce_box_height = coordinates[3]
        face_box_bottomright_x = coordinates[0] + face_box_width
        face_box_bottomright_y = coordinates[1] + foce_box_height

        righteye_x = coordinates[4]
        righteye_y = coordinates[5]

        lefteye_x = coordinates[6]
        lefteye_y = coordinates[7]

        nosetip_x = coordinates[8]
        nosetip_y = coordinates[9]

        mouth_rightcorner_x = coordinates[10]
        mouth_rightcorner_y = coordinates[11]

        mouth_leftcorner_x = coordinates[12]
        mouth_leftcorner_y = coordinates[13]

        # Draw rectangles, lines, and circles on the input image
        cv2.rectangle(input, (face_box_topleft_x, face_box_topleft_y), (face_box_bottomright_x, face_box_bottomright_y), facebox_color, thickness)
        cv2.line(input, (righteye_x, righteye_y), (lefteye_x, lefteye_y), eyeline_color, thickness, lineType=cv2.FILLED)  # Eye line
        cv2.line(input, (mouth_leftcorner_x, mouth_leftcorner_y), (mouth_rightcorner_x, mouth_rightcorner_y), mouthline_color, thickness, lineType=cv2.LINE_4)  # Mouth line
        cv2.circle(input, (nosetip_x, nosetip_y), 1, nosetip_color, thickness)  # Nosetip

if __name__ == '__main__':
    imgpath = "../input_images/crowd.pxhere.com.jpg"
    
    # Initialize FaceDetectorYN with parameters
    nms_threshold = 0.3
    score_threshold = 0.5
    yunet_shape = (320, 320)
    topk = 500

    detector = cv2.FaceDetectorYN.create("../weights/9/face_detection_yunet_2023mar.onnx", "", yunet_shape, score_threshold, nms_threshold, topk)

    img = cv2.imread(imgpath)
    detector.setInputSize((img.shape[1], img.shape[0]))
    faces_and_landmarks = detector.detect(img)

    # Call the visualize function to draw on the image
    visualize(img, faces_and_landmarks)

    cv2.namedWindow("face detections", cv2.WINDOW_NORMAL)
    cv2.imshow("face detections", img)
    cv2.waitKey(0)