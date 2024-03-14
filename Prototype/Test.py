import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model

# tf.compat.v1.disable_eager_execution()

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE = [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Load the trained model architecture from JSON file
with open('Saved-Models/BasicEmotion/basic_emotion_model_0.7614.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model_2 = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model_2.load_weights('Saved-Models/BasicEmotion/basic_emotion_model_0.7614.h5')

# Load the trained model architecture from JSON file
with open('Saved-Models/model_0.7812.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models/model_0.7812.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize Haarcascade face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_grad_cam(model, img_array, class_index):
    # Create a model that maps the input image to the desired layer's output
    grad_model = Model(inputs=model.input, outputs=(model.get_layer('conv2d_2').output, model.output))

    # Compute the gradient of the predicted class with respect to the output feature map of the given layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        predicted_class_output = predictions[:, class_index]  # ASD class index assuming ASD class is the first one
        # print(predicted_class_output)

    grads = tape.gradient(predicted_class_output, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # Compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU on the heatmap
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap


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
    predictions1 = loaded_model.predict(frame)[0]  # Get the first (and only) set of predictions
    predictions2 = loaded_model_2.predict(frame)[0]

    return predictions1, predictions2


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

            frame2 = frame.copy()

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

            eye_min_x = int(mesh_points[33][0])
            eye_max_x = int(mesh_points[133][0])
            eye_min_y = int(mesh_points[159][1])
            eye_max_y = int(mesh_points[145][1])


            if (eye_max_y - eye_min_y) > 10:

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                center_l = np.array([l_cx, l_cy], dtype=np.int32)
                cv2.circle(frame, center_l, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_r = np.array([r_cx, r_cy], dtype=np.int32)
                cv2.circle(frame, center_r, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

                left_pupil_point = (int(mesh_points[468][0]), int(mesh_points[468][1]))
                cv2.circle(frame, left_pupil_point, 3, (0, 0, 255), 1, cv2.LINE_AA)

                right_pupil_point = (int(mesh_points[473][0]), int(mesh_points[473][1]))
                cv2.circle(frame, right_pupil_point, 3, (0, 0, 255), 1, cv2.LINE_AA)

                left_eye_center_x = eye_min_x + ((eye_max_x - eye_min_x) / 2)
                left_eye_center_y = eye_min_y + ((eye_max_y - eye_min_y) / 2)
                left_c = (int(left_eye_center_x), int(left_eye_center_y))

                # cv2.circle(frame, left_c, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(frame, str(left_c[0]), (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # cv2.putText(frame, str(left_pupil_point[0]), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(frame, f'Eye Gaze x-axis: {left_c[0] - left_pupil_point[0]}', (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f'Eye Gaze y-axis: {left_c[1] - left_pupil_point[1]}', (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if 5 < (left_c[0] - left_pupil_point[0]):
                    cv2.putText(frame, "Looking Left", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 0, 0), 2)
                elif 5 < (left_pupil_point[0] - left_c[0]):
                    cv2.putText(frame, "Looking Right", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 0, 0), 2)
                elif 3 < (left_c[1] - left_pupil_point[1]):
                    cv2.putText(frame, "Looking Up", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 0, 0), 2)
                elif 3 < (left_pupil_point[0] - left_c[0]):
                    cv2.putText(frame, "Looking Down", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Blinking", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


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
                eyes_roi = frame2[min(int(landmark_points[104, 1]), int(landmark_points[151, 1]), int(landmark_points[333, 1])):max(int(landmark_points[229, 1]),int(landmark_points[449, 1])),
                            int(landmark_points[162, 0]):int(landmark_points[389, 0])]

                mouth_roi = frame2[int(landmark_points[2, 1]):int(landmark_points[18, 1]),
                            int(landmark_points[215, 0]):int(landmark_points[435, 0])]

                # Resize eyes and mouth regions to a fixed size (adjust as needed)
                eyes_roi = cv2.resize(eyes_roi, (144, 50))
                mouth_roi = cv2.resize(mouth_roi, (144, 30))

                # Combine individual ROIs into a single image
                combined_image = np.vstack((eyes_roi, mouth_roi))

                # Resize the combined image to the desired size (adjust as needed)
                combined_image = cv2.resize(combined_image, (144, 80))

                img = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
                img_2 = img.copy()
                # Normalize the image pixel values to the range [0, 1]
                img = img / 255.0
                # Expand the dimensions of the image to match the input shape expected by the model
                img = np.expand_dims(img, axis=0)

                # plt.figure()
                # plt.imshow(np.squeeze(img))
                # plt.title('Face Region')
                # plt.axis('off')
                # plt.show()

                predictions1, predictions2 = predict_single_frame(img)



                prediction = loaded_model.predict(img)[0][0]  # Access the first element for ASD probability
                # print("prediction: ", prediction)
                # print("prediction: {:.2f}".format(prediction))

                # # Visualize the Grad-CAM heatmap
                # heatmap = generate_grad_cam(loaded_model, img, 0)
                #
                # # Resize heatmap to match the size of the original image
                # heatmap = cv2.resize(heatmap, (144, 80))
                #
                # # Apply colormap for better visualization
                # heatmap = np.uint8(255 * heatmap)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                #
                # heatmap = heatmap.astype(img_2.dtype)
                #
                # # Superimpose the heatmap on the original image
                # superimposed_img = cv2.addWeighted(img_2, 0.6, heatmap, 0.4, 0)
                # superimposed_img = cv2.resize(superimposed_img, (432, 240))
                # cv2.imshow("Grad-CAM", superimposed_img)

                label_text = (f'Emotional State: Boredom={predictions1[0]:.2f}, Engagement={predictions1[1]:.2f}, '
                              f'Confusion={predictions1[2]:.2f}, Frustration={predictions1[3]:.2f}')
                label_text_2 = (f'Basic Emotional State: Happiness={predictions2[0]:.2f}, Surprise={predictions2[1]:.2f}, '
                              f'Sadness={predictions2[2]:.2f}, Anger={predictions2[3]:.2f}, Disgust={predictions2[3]:.2f}')
                cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.putText(frame, label_text_2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

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
                success, rotation_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

                # Get the y rotation degree
                y = angles[0] * 360
                x = angles[1] * 360

                # Add the text on the image
                cv2.putText(frame, "Face Horizontal Angle: " + str(np.round(x, 2)), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Face Vertical Angle:   " + str(np.round(y, 2)), (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)


        cv2.imshow("Engagement Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break
    except Exception as e:
        import traceback

        traceback.print_exc()
        cv2.imshow("Engagement Detection", frame)
        continue
cap.release()
cv2.destroyAllWindows()
