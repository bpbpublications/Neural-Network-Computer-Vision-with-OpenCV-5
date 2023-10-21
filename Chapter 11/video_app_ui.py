import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
import cv2
import numpy as np
import threading
import time

class VideoAppUI:
    def __init__(self, root, improcessor):
        self.__initialize_processing_widgets(root)
        self.__initialize_processing_variables(improcessor)

    def __initialize_processing_widgets(self, root):
        self.__root = root

        # Setting title
        root.title("OpenCV DNN End-to-end Demo")

        # Setting window size
        width = 320
        height = 240
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        # Create Start Video button
        self.__btn_start_video = tk.Button(root)
        self.__btn_start_video["anchor"] = "w"
        self.__btn_start_video["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times', size=10)
        self.__btn_start_video["font"] = ft
        self.__btn_start_video["fg"] = "#000000"
        self.__btn_start_video["justify"] = "center"
        self.__btn_start_video["text"] = "Start Video"
        self.__btn_start_video.place(x=30, y=50, width=70, height=25)
        self.__btn_start_video["command"] = self.__start_video

        # Create Stop Video button
        self.__btn_stop_video = tk.Button(root)
        self.__btn_stop_video["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times', size=10)
        self.__btn_stop_video["font"] = ft
        self.__btn_stop_video["fg"] = "#000000"
        self.__btn_stop_video["justify"] = "center"
        self.__btn_stop_video["text"] = "Stop Video"
        self.__btn_stop_video.place(x=30, y=100, width=70, height=25)
        self.__btn_stop_video["command"] = self.__stop_video
        self.__btn_stop_video["state"] = tk.DISABLED

        # Create Process Video checkbox
        self.__process_video_var = tk.BooleanVar()
        self.__chk_process_video = tk.Checkbutton(root, variable=self.__process_video_var)
        self.__chk_process_video["anchor"] = "w"
        ft = tkFont.Font(family='Times', size=10)
        self.__chk_process_video["font"] = ft
        self.__chk_process_video["fg"] = "#333333"
        self.__chk_process_video["justify"] = "center"
        self.__chk_process_video["text"] = "Process video"
        self.__chk_process_video.place(x=0, y=170, width=150, height=30)
        self.__chk_process_video["offvalue"] = False
        self.__chk_process_video["onvalue"] = True
        self.__chk_process_video["command"] = self.__chk_process_video_command

        # Create Detect Objects checkbox
        self.__detect_objects_var = tk.BooleanVar()
        self.__chk_detect_objects = tk.Checkbutton(root, variable=self.__detect_objects_var)
        ft = tkFont.Font(family='Times', size=10)
        self.__chk_detect_objects["font"] = ft
        self.__chk_detect_objects["fg"] = "#333333"
        self.__chk_detect_objects["justify"] = "center"
        self.__chk_detect_objects["text"] = "Detect objects"
        self.__chk_detect_objects.place(x=10, y=210, width=135, height=30)
        self.__chk_detect_objects["offvalue"] = False
        self.__chk_detect_objects["onvalue"] = True
        self.__chk_detect_objects["state"] = tk.DISABLED

        # Create Detect Number Plate checkbox
        self.__detect_number_plates_var = tk.BooleanVar()
        self.__chk_detect_numberplate = tk.Checkbutton(root, variable=self.__detect_number_plates_var)
        ft = tkFont.Font(family='Times', size=10)
        self.__chk_detect_numberplate["font"] = ft
        self.__chk_detect_numberplate["fg"] = "#333333"
        self.__chk_detect_numberplate["justify"] = "center"
        self.__chk_detect_numberplate["text"] = "Detect Number Plate"
        self.__chk_detect_numberplate.place(x=10, y=250, width=175, height=30)
        self.__chk_detect_numberplate["offvalue"] = False
        self.__chk_detect_numberplate["onvalue"] = True
        self.__chk_detect_numberplate["state"] = tk.DISABLED

        self.__btn_start_video.pack(pady=5)
        self.__btn_stop_video.pack(pady=5)
        self.__chk_process_video.pack(pady=5)
        self.__chk_detect_objects.pack()
        self.__chk_detect_numberplate.pack()

    def __initialize_processing_variables(self, improcessor):
        self.__improcessor = improcessor
        self.__number_of_skipframes = None
        self.__video_capture = None
        self.__video_thread = None

    def __start_video(self):
        self.__video_capture = cv2.VideoCapture(0)
        self.__btn_stop_video["state"] = tk.NORMAL
        self.__btn_start_video["state"] = tk.DISABLED
        self.__video_thread = threading.Thread(target=self.__process_video)
        self.__video_thread.daemon = True
        self.__video_thread.start()

    def __stop_video(self):
        if self.__video_capture:
            self.__video_capture.release()
            self.__video_capture = None  # This will break the infinite loop in __process_video
            self.__video_thread.join()
            self.__btn_stop_video["state"] = tk.DISABLED
            self.__btn_start_video["state"] = tk.NORMAL

    def __chk_process_video_command(self):
        process_video = self.__process_video_var.get()

        if process_video:
            self.__chk_detect_objects["state"] = tk.NORMAL
            self.__chk_detect_numberplate["state"] = tk.NORMAL
        else:
            self.__chk_detect_objects["state"] = tk.DISABLED
            self.__chk_detect_numberplate["state"] = tk.DISABLED

    def __process_video(self):
        if not self.__video_capture:
            messagebox.showerror("Error", "Video is not running. Start the video first.")
            return

        cv2.namedWindow("Video", cv2.WINDOW_FULLSCREEN)
        while True:
            process_video = self.__process_video_var.get()
            detect_objects = self.__detect_objects_var.get()
            detect_number_plates = self.__detect_number_plates_var.get()

            if self.__video_capture:
                ret, frame = self.__video_capture.read()
            else:
                ret = False

            if not ret:
                break

            if process_video:
                if detect_objects:
                    processed_frame = self.__detect_objects(frame)
                else:
                    processed_frame = frame

                if detect_number_plates:
                    processed_frame = self.__detect_number_plate(processed_frame)
            else:
                processed_frame = frame

            # Display the processed frame
            cv2.imshow("Video", processed_frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def __detect_objects(self, frame):
        # Implement YOLOv5 object detection here and mark objects in the frame
        # You'll need to use YOLOv5 and its weights for this part
        # Example code for YOLOv5 detection:
        # result_frame = yolov5_detection(frame)

        result_frame = self.__improcessor.detect_objects(frame)
        return result_frame

    def __detect_number_plate(self, frame):
        result_frame = self.__improcessor.detect_numberplate(frame)
        return result_frame

    def run(self):
        self.__root.mainloop()

