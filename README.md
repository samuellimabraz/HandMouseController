# HandMouseController
This project utilizes MediaPipe and PyAutoGUI to control the mouse cursor and execute actions based on hand gestures. The Hand Gesture Mouse Control allows you to interact with your computer using simple hand movements.

## Prerequisites
- Python 3.8 and up
- The required packages are listed in the requirements.txt file
  
## Installation
1. Clone the repository:
```bash
git clone https://github.com/samuellimabraz/HandMouseController.git
```
2. Change into the project directory:
```bash
cd HandMouseController
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py
```

- The webcam feed will open, and you'll see a bounding box for hand detection.
- Move your index finger within the bounding box to control the mouse cursor.
- Perform gestures to trigger actions:
  - Click: Bring your middle finger close to your index finger.
  - Space bar press: Bring your thumb close to your index finger.
- Press 'q' to exit the application.

## Configuration
You can customize the HandTracker parameters in the main.py file:

```python
tracker = HandTracker(
    model=model,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```
Adjust the confidence values based on your preferences for hand detection and tracking reliability.

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is licensed under the MIT License.

