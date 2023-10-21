import cv2

def detect_face_using_haar(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # convert to gray scale of each frames
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    front_faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)

    # To draw a rectangle in a face
    for (x,y,w,h) in front_faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5)

    return img



# Main script execution
if __name__ == "__main__":
    image_path = "../input_images/crowd.pxhere.com.jpg"

    img = cv2.imread(image_path)
    detected_faces = detect_face_using_haar(img)
    cv2.namedWindow("face detections", cv2.WINDOW_NORMAL)
    cv2.imshow("face detections", detected_faces)
    cv2.waitKey(0)