import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model architecture from JSON file
with open('Saved-Models/model_0.8281.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models/model_0.8281.h5')

# Load the Haarcascades face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to make predictions on a single frame
def predict_single_frame(frame):
    preprocessed_frame = preprocess_input(frame)
    predictions = loaded_model.predict(preprocessed_frame)[0]  # Get the first (and only) set of predictions
    return predictions

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Initialize variables for frame prediction timing

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region
        face = frame[y:y+h, x:x+w]

        img = cv2.resize(face, (112, 112))
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize the image pixel values to the range [-1, 1]
        frame = frame / 255.0
        # Expand the dimensions of the image to match the input shape expected by the model
        img = np.expand_dims(img, axis=0)
        predictions = predict_single_frame(img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        label_text = f'Predictions: Boredom={predictions[0]:.2f}, Engagement={predictions[1]:.2f}, Confusion={predictions[2]:.2f}, Frustration={predictions[3]:.2f}'
        cv2.putText(frame, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Plot the face region
        # plt.figure()
        # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        # plt.title('Face Region')
        # plt.axis('off')
        # plt.show()


    # Display the frame
    cv2.imshow('Video Feed with Predictions', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
