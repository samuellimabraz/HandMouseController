import os
import sys
import urllib.request

import autopy
import cv2
from HandTracker import HandTracker
import numpy as np


class MouseHandController:
    def __init__(self, hand_tracker):
        """
        Initialize a MouseHandController instance.

        Args:
            hand_tracker (HandTracker): Instance of the HandTracker class.

        Returns:
            None
        """
        self.hand_tracker = hand_tracker
        self.wScr, self.hScr = autopy.screen.size()
        self.frameR = 100
        self.smoother = 7
        self.previous_time = 0
        self.previousX, self.previousY = 0, 0
        self.currentX, self.currentY = 0, 0

    def process(self, frame, draw=True) -> np.ndarray:
        """
        Processes the image.

        Args:
            frame (numpy.ndarray): Image to process.
            draw (bool, optional): Whether to draw the landmarks on the image. Defaults to True.

        Returns:
            numpy.ndarray: Image with the landmarks drawn if draw is True, else the original image.
        """
        frame = cv2.flip(frame, 1)
        frame = self.hand_tracker.detect(frame, draw=draw)
        landmarks = self.hand_tracker.get_hand_landmarks()

        if len(landmarks) != 0:
            cv2.rectangle(
                frame,
                (self.frameR, self.frameR),
                (frame.shape[1] - self.frameR, frame.shape[0] - self.frameR),
                (255, 0, 255),
                2,
            )

            x1, y1 = landmarks[8].x * frame.shape[1], landmarks[8].y * frame.shape[0]

            fingers = self.hand_tracker.raised_fingers()
            if fingers[1] == 1:
                x3 = np.interp(
                    x1, (self.frameR, frame.shape[1] - self.frameR), (0, self.wScr)
                )
                y3 = np.interp(
                    y1, (self.frameR, frame.shape[0] - self.frameR), (0, self.hScr)
                )

                self.currentX = self.previousX + (x3 - self.previousX) / self.smoother
                self.currentY = self.previousY + (y3 - self.previousY) / self.smoother

                autopy.mouse.move(self.currentX, self.currentY)

                self.previousX, self.previousY = self.currentX, self.currentY

                # Click mouse (distance between medium and index finger < 28)
                if fingers[2] == 1:
                    lenght, frame, line_info = self.hand_tracker.find_distance(
                        8, 12, frame
                    )
                    if lenght < 28:
                        autopy.mouse.click()
                        cv2.circle(
                            frame,
                            (line_info[4], line_info[5]),
                            15,
                            (0, 255, 0),
                            cv2.FILLED,
                        )

                # Press space (distance between thumb and index finger < 28)
                if fingers[0] == 1:
                    lenght, frame, line_info = self.hand_tracker.find_distance(
                        4, 8, frame
                    )
                    if lenght < 28:
                        autopy.key.tap(autopy.key.Code.SPACE)

        return frame



def main():
    model = HandTracker.download_model()

    tracker = HandTracker(
        model=model,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mouse_hand_controller = MouseHandController(tracker)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        try:
            image = mouse_hand_controller.process(image)
        except Exception as e:
            print(e)
            break

        cv2.imshow("hand_landmarker", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
