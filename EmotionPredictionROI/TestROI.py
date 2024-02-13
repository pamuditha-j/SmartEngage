import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import dlib

# Load the trained model architecture from JSON file
with open('Saved-Models/model_0.8125.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models/model_0.8125.h5')

# Load the Haarcascades face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Dlib shape predictor for 68-point face landmarks
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Provide the correct path to your shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None, None

    landmarks = predictor(gray, faces[0])
    landmarks = landmarks.parts()

    return landmarks, faces[0]


# Function to make predictions on a single frame
def predict_single_frame(frame):
    preprocessed_frame = preprocess_input(frame)
    predictions = loaded_model.predict(preprocessed_frame)[0]  # Get the first (and only) set of predictions
    return predictions


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame)
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[159],  # Left pupil
                                  face_landmarks.landmark[33]]  # Left eye outer corner

            right_eye_landmarks = [face_landmarks.landmark[386],  # Right pupil
                                   face_landmarks.landmark[263]]  # Right eye outer corner

            # Extract the normalized coordinates of left and right eye landmarks
            left_eye_coords = np.array([[lm.x, lm.y] for lm in left_eye_landmarks])
            right_eye_coords = np.array([[lm.x, lm.y] for lm in right_eye_landmarks])

            # Calculate the midpoint of each eye
            left_eye_midpoint = np.mean(left_eye_coords, axis=0)
            right_eye_midpoint = np.mean(right_eye_coords, axis=0)

            # Calculate the direction vector from the midpoint of the left eye to the midpoint of the right eye
            gaze_direction_vector = right_eye_midpoint - left_eye_midpoint

            # Scale the direction vector for visualization
            gaze_direction_vector *= 50  # You can adjust the scaling factor as needed

            # Draw a line representing the gaze direction
            gaze_start_point = (int(left_eye_midpoint[0] * img_w), int(left_eye_midpoint[1] * img_h))
            gaze_end_point = (int(gaze_start_point[0] + gaze_direction_vector[0]),
                              int(gaze_start_point[1] + gaze_direction_vector[1]))
            cv2.arrowedLine(frame, gaze_start_point, gaze_end_point, (0, 255, 0), 2)


            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    n, m = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([n, m])

                    # Get the 3D Coordinates
                    face_3d.append([n, m, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            y = angles[0] * 360
            x = angles[1] * 360

            # Add the text on the image
            cv2.putText(frame, "Horizontal Angle: " + str(np.round(x, 2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Vertical Angle:   " + str(np.round(y, 2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    landmarks, face_rect = get_face_landmarks(frame)

    if landmarks is not None:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        # Extract eyes within the face region
        eye_points = landmarks[36:48]  # Assuming landmarks 36-47 correspond to the eyes
        eyes_x = min(point.x for point in eye_points) - 25
        eyes_y = min(point.y for point in eye_points) - 20
        eyes_w = max(point.x for point in eye_points) - eyes_x + 45
        eyes_h = max(point.y for point in eye_points) - eyes_y + 10

        eyes_roi = cv2.resize(frame[eyes_y:eyes_y + eyes_h, eyes_x:eyes_x + eyes_w], (48, 24))

        mouth_points = landmarks[48:68]  # Assuming landmarks 48-67 correspond to the mouth
        mouth_x = min(point.x for point in mouth_points) - 25
        mouth_y = min(point.y for point in mouth_points) - 15
        mouth_w = max(point.x for point in mouth_points) - mouth_x + 45
        mouth_h = max(point.y for point in mouth_points) - mouth_y + 15

        mouth_roi = cv2.resize(frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (48, 24))

        combined_image = np.hstack((eyes_roi, mouth_roi))

        img = cv2.resize(combined_image, (192, 48))
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize the image pixel values to the range [0, 1]
        img = img / 255.0
        # Expand the dimensions of the image to match the input shape expected by the model
        img = np.expand_dims(img, axis=0)

        # plt.figure()
        # plt.imshow(np.squeeze(img))
        # plt.title('Face Region')
        # plt.axis('off')
        # plt.show()

        predictions = predict_single_frame(img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        label_text = f'Predictions: Boredom={predictions[0]:.2f}, Engagement={predictions[1]:.2f}, Confusion={predictions[2]:.2f}, Frustration={predictions[3]:.2f}'
        cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Combined Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
