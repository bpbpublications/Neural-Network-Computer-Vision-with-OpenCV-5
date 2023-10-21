import tkinter as tk
import video_app_ui as mainui
import image_processor as mainproc

def main():
    # Create a Tkinter root window
    root = tk.Tk()

    # Initialize the ImageProcessor object with required filenames and parameters
    mainprocessor = mainproc.ImageProcessor(
        "weights/8/yolo/YOLOv5s.onnx",  # Object detection model file
        "weights/8/yolo/coco.names",   # Object detection class labels
        "weights/9/text_detection_DB_TD500_resnet18_2021sep.onnx",  # Text detection model file
        "weights/9/text_recognition_CRNN_EN_2021sep.onnx",         # Text recognition model file
        confidence_threshold=0.8  # Confidence threshold for object detection
    )

    # Create the VideoAppUI using the Tkinter root and ImageProcessor instance
    app = mainui.VideoAppUI(root, mainprocessor)

    # Start the Tkinter main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
