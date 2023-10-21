import cv2

def save_frames_as_jpg(video_path, output_directory):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Read and save each frame as a JPG file
    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Save the frame as a JPG file
        output_path = f"{output_directory}/frame_{frame_count}.jpg"
        cv2.imwrite(output_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Frames saved: {frame_count}")

# Example usage:
video_path = "input_images/BigBuckBunny.mp4"
output_directory = "output_images/BigBuckBunny_Frames"

save_frames_as_jpg(video_path, output_directory)
# This program defines a function save_frames_as_jpg that takes the input video path and an output directory as arguments. 
# It uses OpenCV's VideoCapture to read the video file and imwrite to save each frame as a JPG file.
# In the example usage, you need to replace "path/to/input/video.mp4" with the actual path of your input MP4 file, 
# and "path/to/output/directory" with the desired directory to save the individual JPG frames.
# When you run the program, it will read the video file, extract each frame, and save them as individual JPG files in the 
# specified output directory. The program will print the number of frames saved once the process is complete.