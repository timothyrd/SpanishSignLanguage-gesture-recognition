"""
Author: Timothy Ruiz Docena
Date: September 2024

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/


Description:
This code is part of the Spanish Sign Language gesture recognition project.
It captures video from the camera and detects Spanish Sign Languages gestures. It then 
classifies the detected gestures using a CNN pre-trained model in real time.
"""

import cv2
import numpy as np
import keras
from collections import deque
import json
import os
import sys


# Check arguments were passed
if len(sys.argv) > 1:
    # sys.argv is a list where the first element is the script name
    model_path = sys.argv[1]
else:
    raise Exception("No se ha indicado que modelo utilizar.")

# Load the trained model
model = keras.models.load_model(model_path)

# Define the input dimensions expected by the model
frame_width, frame_height = 128, 128  # Size expected by the model
num_frames = 30  # Number of frames in the buffer

# Buffer to store the frames
frame_buffer = deque(maxlen=num_frames)

# Load the gesture dictionary from the JSON file
try:
    with open(os.path.join(__file__,"..",'encodedTimDataset.json'), 'r', encoding='utf-8-sig') as f:
        gesture_dict = json.load(f)
except FileNotFoundError:
    raise Exception("The file was not found.")
except json.JSONDecodeError:
    raise Exception("Error decoding JSON file.")

# Create a mapping from indexes to gesture tags
gesture_map = {np.argmax(encoding): gesture for gesture, encoding in gesture_dict.items()}

# Capture video from camera
cap = cv2.VideoCapture(0)

# Get video resolution
original_width, original_height = int(cap.get(3)), int(cap.get(4))

# Variables for smoothing predictions
prediction_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to greyscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to the expected size by the model
    resized_frame = cv2.resize(gray_frame, (frame_width, frame_height))

    # Normalize the frame
    normalized_frame = resized_frame.astype('float32')  # / 255.0

    # Add a dimension for the channels (necessary to match the expected input)
    normalized_frame = np.expand_dims(normalized_frame, axis=-1)

    # Add the frame to the buffer
    frame_buffer.append(normalized_frame)

    # Make prediction only if the buffer is full
    if len(frame_buffer) == num_frames:
        # Convert the buffer to a numpy array
        frames_array = np.array(frame_buffer)

        # Add batch dimension
        frames_array = np.expand_dims(frames_array, axis=0)  # Shape: (1, 30, 128, 128, 1)

        # Make prediction
        prediction = model.predict(frames_array, verbose=0)

        # Get the index of the predicted class
        predicted_index = np.argmax(prediction, axis=1)[0]

        # Add the prediction to the prediction buffer
        prediction_buffer.append(predicted_index)

        # Obtain the corresponding gesture
        predicted_gesture = gesture_map.get(predicted_index, "IGNORE")

        if np.array(prediction)[0][predicted_index] > 0.85:
            # Show the prediction in the video frame
            cv2.putText(frame, f'Gesto: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

    # Show video in real time
    cv2.imshow('Video en tiempo real', frame)
    k = cv2.waitKey(1)
    # Break the loop with 'q' key
    if k & 0xFF == ord('q'):
        break

# Free up camera and video writer resources, and close the windows
cap.release()
cv2.destroyAllWindows()