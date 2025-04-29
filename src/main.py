import sys
import cv2
import pyautogui
from HandTracker import HandTracker
import numpy as np
import time
from collections import deque


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
        self.wScr, self.hScr = pyautogui.size()
        print(f"Screen size detected: {self.wScr}x{self.hScr}")
        self.frameR = 80
        self.smoother = 5  # Reduced for more responsive movement
        self.previous_time = 0
        self.previousX, self.previousY = 0, 0
        self.currentX, self.currentY = 0, 0
        self.last_space_press = 0
        self.key_cooldown = 1.0  # seconds

        # Performance optimization variables
        self.frame_count = 0
        self.process_every_n_frames = 1  # Process every frame by default
        self.last_cursor_update = 0
        self.cursor_update_interval = (
            0.005  # Reduced interval for more responsive control
        )
        self.last_fps_time = time.time()
        self.fps = 0
        self.fps_frames = 0

        # Debug information
        self.debug_mode = True
        self.cursor_trail = deque(maxlen=10)  # Store recent cursor positions

        # Test cursor movement to ensure it's working
        try:
            print("Testing cursor movement...")
            # Get initial position
            start_x, start_y = pyautogui.position()
            print(f"Initial cursor position: {start_x}, {start_y}")

            # Move cursor to center of screen
            center_x, center_y = self.wScr // 2, self.hScr // 2
            print(f"Moving cursor to center: {center_x}, {center_y}")
            pyautogui.moveTo(center_x, center_y, duration=0.5)

            # Get new position
            new_x, new_y = pyautogui.position()
            print(f"New cursor position: {new_x}, {new_y}")

            # Check if cursor moved
            if abs(new_x - center_x) < 10 and abs(new_y - center_y) < 10:
                print("Cursor movement successful!")
            else:
                print("WARNING: Cursor did not move to the expected position!")

            # Move back to original position
            pyautogui.moveTo(start_x, start_y, duration=0.5)
        except Exception as e:
            print(f"Error during cursor test: {e}")

    def process(self, frame, draw=True) -> np.ndarray:
        """
        Processes the image.

        Args:
            frame (numpy.ndarray): Image to process.
            draw (bool, optional): Whether to draw the landmarks on the image. Defaults to True.

        Returns:
            numpy.ndarray: Image with the landmarks drawn if draw is True, else the original image.
        """
        # FPS calculation
        current_time = time.time()
        self.fps_frames += 1
        if current_time - self.last_fps_time > 1.0:  # Update FPS every second
            self.fps = self.fps_frames / (current_time - self.last_fps_time)
            self.fps_frames = 0
            self.last_fps_time = current_time

        # Always flip the frame first before adding any text or visual elements
        frame = cv2.flip(frame, 1)

        # Draw FPS on screen after flipping
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Display current cursor position after flipping
        curr_x, curr_y = pyautogui.position()
        cv2.putText(
            frame,
            f"Cursor: {curr_x}, {curr_y}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Frame skipping for performance
        self.frame_count += 1
        process_this_frame = self.frame_count % self.process_every_n_frames == 0

        try:
            # Always draw the control area rectangle after flipping
            cv2.rectangle(
                frame,
                (self.frameR, self.frameR),
                (frame.shape[1] - self.frameR, frame.shape[0] - self.frameR),
                (255, 0, 255),
                2,
            )

            # Only run the heavy processing on selected frames
            if process_this_frame:
                # Hand detection (this is the most computationally expensive part)
                frame = self.hand_tracker.detect(frame, draw=draw)
                landmarks = self.hand_tracker.get_hand_landmarks()

                if len(landmarks) != 0 and len(landmarks) >= 21:
                    # Get index finger position
                    x1, y1 = (
                        landmarks[8].x * frame.shape[1],
                        landmarks[8].y * frame.shape[0],
                    )

                    # Draw finger position marker
                    cv2.circle(frame, (int(x1), int(y1)), 10, (0, 0, 255), cv2.FILLED)

                    # Only process gestures if inside the control box
                    if (
                        self.frameR < x1 < frame.shape[1] - self.frameR
                        and self.frameR < y1 < frame.shape[0] - self.frameR
                    ):
                        # Draw active indicator - now also after flipping
                        cv2.putText(
                            frame,
                            "ACTIVE",
                            (frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                        # Get raised fingers status (computationally expensive)
                        fingers = self.hand_tracker.raised_fingers()

                        # Check if fingers list has expected values
                        if (
                            len(fingers) >= 3
                        ):  # We need at least index, middle, and thumb
                            # Only update cursor if enough time has passed since last update
                            if (
                                fingers[1] == 1  # Index finger raised
                                and current_time - self.last_cursor_update
                                >= self.cursor_update_interval
                            ):
                                # Map finger coordinates to screen coordinates
                                # Flip the Y axis mapping to match natural movement
                                x3 = np.interp(
                                    x1,
                                    (self.frameR, frame.shape[1] - self.frameR),
                                    (0, self.wScr),
                                )
                                y3 = np.interp(
                                    y1,
                                    (self.frameR, frame.shape[0] - self.frameR),
                                    (0, self.hScr),
                                )

                                # Smooth the movement
                                self.currentX = int(
                                    self.previousX
                                    + (x3 - self.previousX) / self.smoother
                                )
                                self.currentY = int(
                                    self.previousY
                                    + (y3 - self.previousY) / self.smoother
                                )

                                # Keep cursor within screen bounds
                                self.currentX = max(
                                    0, min(self.wScr - 1, self.currentX)
                                )
                                self.currentY = max(
                                    0, min(self.hScr - 1, self.currentY)
                                )

                                # Debug visualization
                                if self.debug_mode:
                                    # Convert screen coordinates back to camera space for visualization
                                    cam_x = int(
                                        np.interp(
                                            self.currentX,
                                            (0, self.wScr),
                                            (self.frameR, frame.shape[1] - self.frameR),
                                        )
                                    )
                                    cam_y = int(
                                        np.interp(
                                            self.currentY,
                                            (0, self.hScr),
                                            (self.frameR, frame.shape[0] - self.frameR),
                                        )
                                    )

                                    # Add to cursor trail
                                    self.cursor_trail.append((cam_x, cam_y))

                                    # Draw cursor trail
                                    for i in range(1, len(self.cursor_trail)):
                                        cv2.line(
                                            frame,
                                            self.cursor_trail[i - 1],
                                            self.cursor_trail[i],
                                            (0, 255, 255),
                                            2,
                                        )

                                    # Draw current cursor position
                                    if self.cursor_trail:
                                        cv2.circle(
                                            frame,
                                            self.cursor_trail[-1],
                                            8,
                                            (0, 255, 255),
                                            cv2.FILLED,
                                        )

                                # Try both direct control methods
                                try:
                                    # Method 1: Standard PyAutoGUI moveTo
                                    pyautogui.moveTo(
                                        self.currentX, self.currentY, _pause=False
                                    )

                                    # For debugging
                                    if (
                                        self.frame_count % 30 == 0
                                    ):  # Log less frequently
                                        print(
                                            f"Moving cursor to: {self.currentX}, {self.currentY}"
                                        )
                                        print(
                                            f"Actual cursor position: {pyautogui.position()}"
                                        )

                                except Exception as e:
                                    print(f"Error moving cursor: {e}")

                                self.last_cursor_update = current_time
                                self.previousX, self.previousY = (
                                    self.currentX,
                                    self.currentY,
                                )

                                # Click mouse (distance between medium and index finger < 28)
                                if fingers[2] == 1:  # Middle finger raised
                                    try:
                                        length, frame, line_info = (
                                            self.hand_tracker.find_distance(
                                                8, 12, frame
                                            )
                                        )
                                        if length < 35:
                                            # Use pyautogui for mouse click
                                            pyautogui.click()
                                            print("Click!")
                                            cv2.circle(
                                                frame,
                                                (line_info[4], line_info[5]),
                                                15,
                                                (0, 255, 0),
                                                cv2.FILLED,
                                            )
                                    except (IndexError, ValueError) as e:
                                        print(
                                            f"Error measuring finger distance for click: {e}"
                                        )

                                # Press space (distance between thumb and index finger < 28)
                                if fingers[0] == 1:  # Thumb raised
                                    try:
                                        length, frame, line_info = (
                                            self.hand_tracker.find_distance(4, 8, frame)
                                        )
                                        if (
                                            length < 28
                                            and (current_time - self.last_space_press)
                                            > self.key_cooldown
                                        ):
                                            # Use pyautogui for keypress
                                            pyautogui.press("space")
                                            print("Space!")
                                            self.last_space_press = current_time
                                    except (IndexError, ValueError) as e:
                                        print(
                                            f"Error measuring finger distance for space: {e}"
                                        )

                        # If we're in the control box but dropping frames, increase the frame skip
                        if self.fps < 15 and self.process_every_n_frames < 3:
                            self.process_every_n_frames += 1
                            print(
                                f"Performance optimization: Processing every {self.process_every_n_frames} frames"
                            )
                        # If we have good performance, decrease the frame skip
                        elif self.fps > 25 and self.process_every_n_frames > 1:
                            self.process_every_n_frames -= 1
                            print(
                                f"Performance optimization: Processing every {self.process_every_n_frames} frames"
                            )
        except Exception as e:
            print(f"Error in process method: {e}")
            # Do not re-raise the exception, just log it and continue

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
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH, 640
    )  # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # PyAutoGUI settings - removing all delays
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    # Test cursor control on startup
    try:
        print("Testing initial cursor control...")
        start_pos = pyautogui.position()
        test_pos = (100, 100)
        pyautogui.moveTo(test_pos)
        actual_pos = pyautogui.position()
        print(f"Moved to {test_pos}, actual position: {actual_pos}")

        # Move back
        pyautogui.moveTo(start_pos)
    except Exception as e:
        print(f"Error in initial cursor test: {e}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        try:
            image = mouse_hand_controller.process(image)
        except Exception as e:
            print(f"Error processing frame: {e}")  
            break

        cv2.imshow("hand_landmarker", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
