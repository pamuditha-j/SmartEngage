import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import model_from_json

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE = [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Load the trained model architecture from JSON file
with open('Saved-Models/model_0.8438.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models/model_0.8438.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize Haarcascade face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(image_path, face_cascade):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haarcascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assume there's only one face, take the first one
        x, y, w, h = faces[0]
        face_region = image[y:y + h, x:x + w]
        return face_region, (x, y, w, h)
    else:
        return None, None


# Function to make predictions on a single frame
def predict_single_frame(frame):
    predictions = loaded_model.predict(frame)[0]  # Get the first (and only) set of predictions
    return predictions


# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    try:
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haarcascade
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # x, y, w, h = faces[0]
        # face_region = frame[y:y + h, x:x + w]

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            center_l = np.array([l_cx, l_cy], dtype=np.int32)
            cv2.circle(frame, center_l, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_r = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_r, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

            # Convert landmarks to format accepted by cv2.polylines()
            face_contour_points = [(int(point[0]), int(point[1])) for point in mesh_points[FACE]]
            face_contour_points = np.array([face_contour_points], dtype=np.int32)
            # Draw polyline connecting face contour landmarks
            cv2.polylines(frame, [face_contour_points], isClosed=True, color=(0, 255, 0), thickness=1)

            # Convert landmarks to format accepted by cv2.polylines()
            left_eye_contour_points = [(int(point[0]), int(point[1])) for point in mesh_points[LEFT_EYE]]
            left_eye_contour_points = np.array([left_eye_contour_points], dtype=np.int32)
            # Draw polyline connecting face contour landmarks
            cv2.polylines(frame, [left_eye_contour_points], isClosed=True, color=(0, 255, 0), thickness=1)

            # Convert landmarks to format accepted by cv2.polylines()
            right_eye_contour_points = [(int(point[0]), int(point[1])) for point in mesh_points[RIGHT_EYE]]
            right_eye_contour_points = np.array([right_eye_contour_points], dtype=np.int32)
            # Draw polyline connecting face contour landmarks
            cv2.polylines(frame, [right_eye_contour_points], isClosed=True, color=(0, 255, 0), thickness=1)

            left_pupil_point = (int(mesh_points[468][0]), int(mesh_points[468][1]))
            cv2.circle(frame, left_pupil_point, 3, (0, 0, 255), 1, cv2.LINE_AA)

            right_pupil_point = (int(mesh_points[473][0]), int(mesh_points[473][1]))
            cv2.circle(frame, right_pupil_point, 3, (0, 0, 255), 1, cv2.LINE_AA)

            eye_min_x = int(mesh_points[33][0])
            eye_max_x = int(mesh_points[133][0])
            eye_min_y = int(mesh_points[159][1])
            eye_max_y = int(mesh_points[145][1])

            left_eye_center_x = eye_min_x + ((eye_max_x - eye_min_x) / 2)
            left_eye_center_y = eye_min_y + ((eye_max_y - eye_min_y) / 2)
            left_c = (int(left_eye_center_x), int(left_eye_center_y))

            # cv2.circle(frame, left_c, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(frame, str(left_c[0]), (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # cv2.putText(frame, str(left_pupil_point[0]), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f'Eye Gaze x-axis: {left_c[0] - left_pupil_point[0]}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'Eye Gaze y-axis: {left_c[1] - left_pupil_point[1]}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if 5 < (left_c[0] - left_pupil_point[0]):
                cv2.putText(frame, "Looking Left", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif 5 < (left_pupil_point[0] - left_c[0]):
                cv2.putText(frame, "Looking Right", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif 3 < (left_c[1] - left_pupil_point[1]):
                cv2.putText(frame, "Looking Up", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif 3 < (left_pupil_point[0] - left_c[0]):
                cv2.putText(frame, "Looking Down", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


            for face_landmarks in results.multi_face_landmarks:

                # Extract landmark points
                landmark_points = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

                # Convert the landmark points to NumPy array
                landmark_points = np.array(landmark_points)

                # Calculate bounding box coordinates
                min_x = int(np.min(landmark_points[:, 0]))
                max_x = int(np.max(landmark_points[:, 0]))
                min_y = int(np.min(landmark_points[:, 1]))
                max_y = int(np.max(landmark_points[:, 1]))

                # Draw rectangle around the face
                # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                # Extract eyes region
                eyes_roi = frame[int(landmark_points[109, 1]):int(landmark_points[101, 1]),
                           int(landmark_points[162, 0]):int(landmark_points[389, 0])]

                mouth_roi = frame[int(landmark_points[36, 1]):int(landmark_points[201, 1]),
                            int(landmark_points[207, 0]):int(landmark_points[427, 0])]

                face_roi = frame[int(landmark_points[10, 1]):int(landmark_points[152, 1]),
                           int(landmark_points[234, 0]):int(landmark_points[356, 0])]

                # Resize eyes and mouth regions to a fixed size (adjust as needed)
                eyes_roi = cv2.resize(eyes_roi, (144, 72))
                mouth_roi = cv2.resize(mouth_roi, (144, 72))
                face_roi = cv2.resize(face_roi, (244, 244))

                # Combine individual ROIs into a single image
                combined_image = np.hstack((eyes_roi, mouth_roi))

                # Resize the combined image to the desired size (adjust as needed)
                combined_image = cv2.resize(combined_image, (288, 72))

                img = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
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

                label_text = f'Emotional State: Boredom={predictions[0]:.2f}, Engagement={predictions[1]:.2f}, Confusion={predictions[2]:.2f}, Frustration={predictions[3]:.2f}'
                cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
                cv2.putText(frame, "Face Horizontal Angle: " + str(np.round(x, 2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Face Vertical Angle:   " + str(np.round(y, 2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

        cv2.imshow("Combined Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break
    except Exception as e:
        print("Error ", e)
        cv2.imshow("Combined Detection", frame)
        continue
cap.release()
cv2.destroyAllWindows()
