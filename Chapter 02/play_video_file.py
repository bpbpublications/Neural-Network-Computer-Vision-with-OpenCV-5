import cv2

def play_video(filename):
    # Open the video file
    video = cv2.VideoCapture(filename)

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not read successfully, exit the loop
        if not ret:
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    video.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    # Provide the path to your MP4 file
    video_file = 'input_images/BigBuckBunny.mp4'

    # Call the function to play the video
    play_video(video_file)