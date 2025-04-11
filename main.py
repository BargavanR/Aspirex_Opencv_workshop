# Import required libraries
import os  # For handling file paths and system operations

# Import TensorFlow/Keras components
from tensorflow.keras.models import load_model  # To load our pre-trained emotion detection model
from time import sleep  # For creating delays if needed in the detection loop
from tensorflow.keras.preprocessing.image import img_to_array  # Converts images to arrays that our model can process
from tensorflow.keras.preprocessing import image  # Keras utilities for image preprocessing
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations on arrays

# Load the pre-trained face detection model (Haar Cascade)
# This XML file contains the pre-trained model that can identify human faces in images
face_classifier = cv2.CascadeClassifier(r'C:\Users\Bargavan R\OneDrive\Desktop\OPENCV\cnn_example\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

# Load our pre-trained emotion classification model
# This H5 file contains the CNN model architecture and weights trained to recognize emotions
classifier = load_model(r'C:\Users\Bargavan R\OneDrive\Desktop\OPENCV\cnn_example\Emotion_Detection_CNN-main\model.h5')

# Define the emotion labels that correspond to the model's output classes
# The index position of each emotion matches the output neuron of our model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam capture (0 indicates default camera)
cap = cv2.VideoCapture(0)

# Start the continuous detection loop
while True:
    # Read a frame from the webcam
    # The underscore ignores the return status (success/failure) of cap.read()
    _, frame = cap.read()
    labels = []
    
    # Convert the color frame to grayscale
    # Face detection works better on grayscale images and requires less processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    # Returns a list of rectangles (x, y, width, height) where faces were detected
    faces = face_classifier.detectMultiScale(gray)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Extract just the face area (region of interest)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the face to 48x48 pixels as expected by our emotion model
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if we actually have face data (not an empty array)
        if np.sum([roi_gray]) != 0:
            # Preprocess the face image for the model:
            # 1. Convert pixel values to float and normalize to range 0-1
            roi = roi_gray.astype('float') / 255.0
            
            # 2. Convert the image to an array format suitable for Keras
            roi = img_to_array(roi)
            
            # 3. Add a batch dimension (model expects shape: [batch_size, height, width, channels])
            roi = np.expand_dims(roi, axis=0)

            # Make prediction with our model
            # Returns an array of probabilities for each emotion class
            prediction = classifier.predict(roi)[0]
            
            # Find the emotion with highest probability using argmax
            label = emotion_labels[prediction.argmax()]
            
            # Set the position to display the emotion label (top of the face rectangle)
            label_position = (x, y)
            
            # Add the emotion text to the image
            # Parameters: image, text, position, font, scale, color (BGR), thickness
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If no valid face data is found, display 'No Faces' message
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame with all our annotations
    cv2.imshow('Emotion Detector', frame)
    
    # Check for the 'q' key press to exit the loop
    # waitKey(1) waits for 1ms between frames, & 0xFF == ord('q') checks if 'q' was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows when finished
cap.release()
cv2.destroyAllWindows()