import numpy as np
import cv2

if __name__ == '__main__':
    # Paths to the input images
    img1path = "../input_images/GopiKrishnaNuti1.jpg"
    img2path = "../input_images/GopiKrishnaNuti2.jpg"

    # Initialize FaceDetectorYN with parameters
    nms_threshold = 0.3
    score_threshold = 0.5
    yunet_shape = (320, 320)
    topk = 500

    detector = cv2.FaceDetectorYN.create("../weights/9/face_detection_yunet_2023mar.onnx", "", yunet_shape, score_threshold, nms_threshold, topk)

    # Load and detect faces in the first image
    img1 = cv2.imread(img1path)
    detector.setInputSize((img1.shape[1], img1.shape[0]))
    faces_and_landmarks1 = detector.detect(img1)

    # Load and detect faces in the second image
    img2 = cv2.imread(img2path)
    detector.setInputSize((img2.shape[1], img2.shape[0]))
    faces_and_landmarks2 = detector.detect(img2)

    # Initialize FaceRecognizerSF
    recognizer = cv2.FaceRecognizerSF.create("../weights/9/face_recognition_sface_2021dec.onnx", "")

    # Align and crop faces from the images
    face1_align = recognizer.alignCrop(img1, faces_and_landmarks1[1][0])
    face2_align = recognizer.alignCrop(img2, faces_and_landmarks2[1][0])

    # Extract features from the aligned faces
    facial_features1 = recognizer.feature(face1_align)
    facial_features2 = recognizer.feature(face2_align)

    # Set similarity score thresholds
    cosine_similarity_threshold = 0.363
    l2_similarity_threshold = 1.128

    # Calculate cosine and L2 similarity scores
    cosine_score = recognizer.match(facial_features1, facial_features2, cv2.FaceRecognizerSF_FR_COSINE)
    l2_score = recognizer.match(facial_features1, facial_features2, cv2.FaceRecognizerSF_FR_NORM_L2)

    # Determine if the images belong to the same person based on similarity scores
    if (cosine_score >= cosine_similarity_threshold) or (l2_score <= l2_similarity_threshold):
        print("Images are of the same person")
    else:
        print("Images do not belong to the same person")
