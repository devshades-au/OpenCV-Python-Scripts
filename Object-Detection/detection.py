"""
A simple object detection and alerting script using OpenCV and COCO's dataset.
@Version: 1.0
@Author: Devshades
@Status: Prototype / Staging
"""

import cv2
import os
from datetime import datetime
from gtts import gTTS
import pyttsx3
import time

class ObjectDetectionConfig:
    """
    Central configuration class for object detection parameters and variables.
    """
    def __init__(self):
        # Object detection parameters
        self.thres = 0.65
        self.nms = 0.2
        self.objects = ["person", "cup"]

        # Video capture configuration
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(3, 640)
        self.video_capture.set(4, 480)

        # List of objects to be alerted about
        self.alert_objects = ['person']

        # Initialize object counts and detection timestamps
        self.object_counts = {obj: 0 for obj in self.objects}
        self.detection_timestamps = {obj: None for obj in self.objects}

class Model:
    """
    Class representing the model data sets for object detection.

    Attributes:
        data_directory (str): Path where the model data is stored.
        classNames (list): List of class names for detected objects.
    """
    data_directory = "data/"
    classNames = []

    # Read class names from a file
    classFile = os.path.join(data_directory, 'coco.names')
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    # Model paths
    configPath = os.path.join(data_directory, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    weightsPath = os.path.join(data_directory, "frozen_inference_graph.pb")

    # Initialize the model
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

class Detection:
    """
    Class for object detection in images.

    Methods:
        Objects(img, thres, nms, draw=True, objects=[]):
            Detect objects in an image and optionally draw bounding boxes and labels.
    """
    @staticmethod
    def Objects(img, thres, nms, draw=True, objects=[]):
        """
        Detect objects in an image and optionally draw bounding boxes and labels.

        Args:
            img (numpy.ndarray): Input image for object detection.
            thres (float): Confidence threshold for object detection.
            nms (float): Non-maximum suppression threshold to filter overlapping boxes.
            draw (bool): If True, draw bounding boxes and labels on the image.
            objects (list): List of specific objects to detect; if empty, detects all objects.

        Returns:
            img (numpy.ndarray): Image with bounding boxes and labels (if draw is True).
            objectInfo (list): List of detected objects and their bounding boxes.
        """
        classIds, confs, bbox = Model.net.detect(img, confThreshold=thres, nmsThreshold=nms)
        objectInfo = []

        if len(objects) == 0:
            objects = Model.classNames

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = Model.classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, Model.classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        return img, objectInfo

def object_detected_alert(object_name):
    """
    Print an alert when a specific object is detected in the frame and trigger TTS alert.

    Args:
        object_name (str): Name of the specific object to check for.

    Returns:
        None
    """
    global config
    current_time = datetime.now().strftime("%d/%m/%y - %H:%M:%S")
    alert_message = "\u001b[34m"+ f" [ 🔍 ] Detection: {object_name.upper()} at {current_time}" + "\u001b[0m"
    print(alert_message)
    time.sleep(1)  # If the feed becomes too laggy - remove or comment this line

    # Check if enough time has passed since the last TTS alert (1 minute)
    if config.detection_timestamps.get(object_name) is None or \
            (time.time() - config.detection_timestamps[object_name] >= 60):
        config.detection_timestamps[object_name] = time.time()
        
        # Check if the object should be alerted about
        if object_name in config.alert_objects:
            speak_alert(f"Detected {object_name}")
            print("\u001b[33m"+f" [ ⚠️ ] Alert: {object_name.upper()} at {current_time}" + "\u001b[0m")

def speak_alert(alert_message):
    """
    Use TTS to speak the alert message.

    Args:
        alert_message (str): The alert message to be spoken.

    Returns:
        None
    """
    try:
        # Use pyttsx3 for text-to-speech
        engine = pyttsx3.init()
        engine.say(alert_message)
        engine.runAndWait()
    except Exception as e:
        print(f"Error while speaking alert: {e}")

if __name__ == "__main__":
    """
    Main script for object detection and labeling using OpenCV and COCO's dataset.
    """
    global config
    config = ObjectDetectionConfig()

    while True:
        success, img = config.video_capture.read()
        result, objectInfo = Detection.Objects(img, thres=config.thres, nms=config.nms, objects=config.objects)

        for _, obj_name in objectInfo:
            if obj_name in config.object_counts:
                config.object_counts[obj_name] += 1

                # Check for a specific object ("person") and print an alert - empty = all objects
                if obj_name in config.objects:
                    object_detected_alert(obj_name)

        detections_label = f"Detecting: {config.objects}" 
        alerts_label = f"Alerting: {config.alert_objects}"
        cam_label = f"CAM-01"
        dt_label = datetime.now().strftime("%d/%m/%y - %H:%M:%S")
        cv2.putText(img, detections_label, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
        cv2.putText(img, alerts_label, (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
        cv2.putText(img, cam_label, (10,400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)
        cv2.putText(img, dt_label, (10, 425), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)  

        cv2.imshow("Live Feed", img)
        cv2.waitKey(1)
