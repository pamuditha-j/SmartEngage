import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the trained model architecture from JSON file
with open('Saved-Models\model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models\model.h5')

# Load the Haarcascades face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess an image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (112, 112))  # Assuming the model expects images of size (112, 112)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to detect and crop the face from an image
def detect_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Assume the first detected face
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    else:
        return None

# Function to make predictions on a single frame
def predict_single_frame(frame):
    preprocessed_frame = preprocess_image(frame)
    predictions = loaded_model.predict(preprocessed_frame)[0]  # Get the first (and only) set of predictions
    return predictions

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect and crop the face
    cropped_face = detect_and_crop_face(frame)

    if cropped_face is not None:
        # Make predictions on the cropped face
        predictions = predict_single_frame(cropped_face)

        # Display predicted class labels on the video feed
        label_text = f'Predictions: Boredom={predictions[0]:.2f}, Engagement={predictions[1]:.2f}, Confusion={predictions[2]:.2f}, Frustration={predictions[3]:.2f}'
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video Feed with Predictions', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
