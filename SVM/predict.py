"""
Author: Timothy Ruiz Docena
Date: September 2024

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/


Description:
This code is part of the Spanish Sign Language gesture recognition project.
It captures video from the camera and detects hand gestures and poses from the Spanish 
Sign Language using MediaPipe. It then classifies the detected gestures using a SVM
pre-trained model in real time.
"""

# Imports
import cv2
import pandas as pd
import pickle
import numpy as np
import mediapipe as mp

from collections import deque
import sys


# Check that arguments were passed
if len(sys.argv) > 1:
    # sys.argv is a list where the first element is the script name
    model_path = sys.argv[1]
else:
    raise Exception("It has not been specified which model to use")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Sequence buffer for gesture recognition
SEQUENCE_LENGTH = 20  # Adjust based on the length of your gestures
sequence = deque(maxlen=SEQUENCE_LENGTH)

# Variables to store the last prediction
last_gesture_class = "Waiting..."
last_gesture_prob = None

display_frames = 0  # Frames to keep the gesture displayed

# Display duration in frames
DISPLAY_DURATION = 30  # Keep the text visible for 30 frames

# Capture video from camera
cap = cv2.VideoCapture(0)

# Initialize holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        results = holistic.process(frame)

        # Make image writable
        frame.flags.writeable = True

        # Left hand (red)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        # Right hand (green)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))

        # Pose
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark[:23]
            for i, landmark in enumerate(pose_landmarks):
                if i >= 23:
                    break
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = int(landmark.z * frame.shape[1])
                cv2.circle(frame, (x, y), 3, (128, 0, 255), -1)
        try:
            # Add coords to list row
            row = []

            # Get right hand landmarks
            for landmark in results.right_hand_landmarks.landmark:
                try:
                    row.extend([landmark.x, landmark.y, landmark.z])
                except:
                    row.extend([0.0, 0.0, 0.0] * len(mp_holistic.HandLandmark))

            # Get left hand landmarks
            for landmark in results.left_hand_landmarks.landmark:
                try:
                    row.extend([landmark.x, landmark.y, landmark.z])
                except:
                    row.extend([0.0, 0.0, 0.0] * len(mp_holistic.HandLandmark))

            # Get pose landmarks
            for i in range(23):
                try:
                    landmark = results.pose_landmarks.landmark[i]
                    row.extend([landmark.x, landmark.y, landmark.z])
                except:
                    row.extend([0.0, 0.0, 0.0] * 23)

            # Store coordinates into the sequence buffer
            sequence.append(row)


        except:
            pass

        # Display the buffer status
        buffer_status = f'Buffer: {len(sequence)}/{SEQUENCE_LENGTH}'
        cv2.putText(frame, buffer_status,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # If the sequence is filled, make a prediction
        if len(sequence) == SEQUENCE_LENGTH:  
            # Convert sequence buffer into a DataFrame
            X = pd.DataFrame(sequence) 

            # Pass previous buffer into the model to make a prediction
            prediction = model.predict(X)

            # Count the frequency of each prediction, the most common gesture is chosen
            counter = {}
            for pred in prediction:
                if pred not in counter:
                    counter[pred] = 0

                counter[pred] += 1
            # Most common gesture is chosen
            most_likely_gesture = max(counter, key=counter.get)

            # Save predicted gesture and its probability
            last_gesture_class = most_likely_gesture
            last_gesture_prob = model.predict_proba(X)[0]

            # Start the display duration timer
            display_frames = DISPLAY_DURATION

            # Clear the buffer after making a prediction
            sequence.clear()

        # Show results
        if display_frames > 0:
            coords = (10, 100)  # Adjusted for better readability

            # Predicted gesture
            cv2.putText(frame, f'Gesto: {last_gesture_class}', coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Probability
            prob_text = f'Probabilidad: {round(last_gesture_prob[np.argmax(last_gesture_prob)], 2)}'
            cv2.putText(frame, prob_text, (coords[0], coords[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Decrease the display frame counter
            display_frames -= 1

        # Show image in real time
        cv2.imshow("Imagen a detectar", frame)
        # Break the loop with 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Free up camera and video writer resources, and close the windows
cap.release()
cv2.destroyAllWindows()