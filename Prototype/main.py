from flask import Flask, render_template, Response, jsonify, make_response
from flask_cors import CORS, cross_origin
import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image as imageX


app = Flask(__name__)
#handles cors error
CORS(app, supports_credentials=True)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE = [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

response_values = ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']

video = cv2.VideoCapture(0)

# load json and create model
json_file = open('TrainedModels/BasicEmotion/prediction_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
basic_emotion_model = model_from_json(loaded_model_json)  # convert the jason into the model
# load weights in h5 file into new model
basic_emotion_model.load_weights("TrainedModels/BasicEmotion/prediction_model.h5")

# load json and create model
json_file_2 = open('TrainedModels/AffectiveState/prediction_model.json', 'r')
loaded_model_json_2 = json_file_2.read()
json_file_2.close()
affective_state_model = model_from_json(loaded_model_json_2)  # convert the jason into the model
# load weights in h5 file into new model
affective_state_model.load_weights("TrainedModels/AffectiveState/prediction_model.h5")

# def calculate_final_engagement_score(eye_gaze_horizontal, eye_gaze_vertical, ):


def predict_single_frame(frame1,frame2):
    predictions1 = basic_emotion_model.predict(frame1)[0]  # Get the first (and only) set of predictions
    predictions2 = affective_state_model.predict(frame2)[0]

    return predictions1, predictions2

def output_frame(video_feed):
    success, frame = video_feed.read()
    while (success):
        frame = cv2.flip(frame, 1)

        results = face_mesh.process(frame)
        img_h, img_w, img_c = frame.shape

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

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

            eyes_top_left = (
            int(mesh_points[162, 0]), min(int(mesh_points[104, 1]), int(mesh_points[151, 1]), int(mesh_points[333, 1])))
            eyes_bottom_right = (int(mesh_points[389, 0]), max(int(mesh_points[229, 1]), int(mesh_points[449, 1])))

            # Coordinates for mouth ROI
            mouth_top_left = (int(mesh_points[215, 0]), int(mesh_points[2, 1]))
            mouth_bottom_right = (int(mesh_points[435, 0]), int(mesh_points[18, 1]))

            # Drawing rectangles around the ROIs
            cv2.rectangle(frame, eyes_top_left, eyes_bottom_right, (255, 0, 0), 2)  # Green rectangle for eyes ROI
            cv2.rectangle(frame, mouth_top_left, mouth_bottom_right, (255, 0, 0), 2)  # Blue rectangle for mouth ROI

            # Adding labels
            cv2.putText(frame, "roi1", (eyes_top_left[0], eyes_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.putText(frame, "roi2", (mouth_top_left[0], mouth_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        success, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def detect_engagement_factors(video_feed):
    while video_feed.isOpened():
        success, frame = video_feed.read()

        image = cv2.flip(frame, 1)
        results = face_mesh.process(frame)
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            eye_min_x = int(mesh_points[33][0])
            eye_max_x = int(mesh_points[133][0])
            eye_min_y = int(mesh_points[159][1])
            eye_max_y = int(mesh_points[145][1])

            pupil_point = (int(mesh_points[468][0]), int(mesh_points[468][1]))

            if (eye_max_y - eye_min_y) > 10:
                left_eye_center_x = eye_min_x + ((eye_max_x - eye_min_x) / 2)
                left_eye_center_y = eye_min_y + ((eye_max_y - eye_min_y) / 2)
                left_c = (int(left_eye_center_x), int(left_eye_center_y))

                eye_gaze_horizontal = f'{left_c[0] - pupil_point[0]}'
                eye_gaze_vertical = f'{left_c[1] - pupil_point[1]}'
                if 5 < (left_c[0] - pupil_point[0]):
                    eye_gaze_text = "Looking Right"
                elif 5 < (pupil_point[0] - left_c[0]):
                    eye_gaze_text = "Looking Left"
                elif 3 < (left_c[1] - pupil_point[1]):
                    eye_gaze_text = "Looking Up"
                elif 3 < (pupil_point[0] - left_c[0]):
                    eye_gaze_text = "Looking Down"
                else:
                    eye_gaze_text = "Looking Straight"
            else:
                eye_gaze_text = "Blinking"
                eye_gaze_vertical = "N/A"
                eye_gaze_horizontal = "N/A"

            # Extract eyes region
            eyes_roi_model1 = frame[min(int(mesh_points[104, 1]), int(mesh_points[151, 1]),
                                  int(mesh_points[333, 1])):max(int(mesh_points[229, 1]),
                                                                    int(mesh_points[449, 1])),
                       int(mesh_points[162, 0]):int(mesh_points[389, 0])]

            mouth_roi_model1 = frame[int(mesh_points[2, 1]):int(mesh_points[18, 1]),
                        int(mesh_points[215, 0]):int(mesh_points[435, 0])]

            # Resize eyes and mouth regions to a fixed size (adjust as needed)
            eyes_roi_model1 = cv2.resize(eyes_roi_model1, (144, 80))
            mouth_roi_model1 = cv2.resize(mouth_roi_model1, (144, 80))

            # Combine individual ROIs into a single image
            combined_image_model1 = np.hstack((eyes_roi_model1, mouth_roi_model1))

            # Resize the combined image to the desired size (adjust as needed)
            combined_image_model1 = cv2.resize(combined_image_model1, (144, 80))

            # img_model1 = cv2.cvtColor(combined_image_model1, cv2.COLOR_BGR2RGB)
            img_model1 = combined_image_model1.astype('float64') / 255.0
            img_model1 = np.expand_dims(img_model1, axis=0)

            eyes_roi_model2 = image[
                       min(int(mesh_points[67, 1]), int(mesh_points[10, 1]), int(mesh_points[297, 1])):max(
                           int(mesh_points[118, 1]), int(mesh_points[346, 1])),
                       int(mesh_points[162, 0]):int(mesh_points[389, 0])]

            mouth_roi_model2 = image[int(mesh_points[4, 1]):int(mesh_points[200, 1]),
                        int(mesh_points[132, 0]):int(mesh_points[361, 0])]

            eyes_roi_model2 = cv2.resize(eyes_roi_model2, (48, 24))
            mouth_roi_model2 = cv2.resize(mouth_roi_model2, (48, 24))

            # Combine individual ROIs into a single image
            combined_image_model2 = np.hstack((eyes_roi_model2, mouth_roi_model2))

            # Resize the combined image to the desired size (adjust as needed)
            combined_image_model2 = cv2.resize(combined_image_model2, (96, 24))
            # print(combined_image_model2.shape)
            img_model2 = cv2.cvtColor(combined_image_model2, cv2.COLOR_BGR2GRAY)
            # print(img_model2.shape)
            # img_model2 = img_model2.astype('float64') / 255.0
            # print(img_model2.shape)
            # img_model2 = np.expand_dims(img_model2, axis=0)

            img_model2 = imageX.img_to_array(img_model2)
            img_model2 = np.expand_dims(img_model2, axis=0)
            img_model2 /= 255.0

            basic_emotion_pred, affective_state_pred = predict_single_frame(img_model2, img_model1)

            affective_state_text = f'Boredom={affective_state_pred[0]:.2f}, Engagement={affective_state_pred[1]:.2f}, Confusion={affective_state_pred[2]:.2f}, Frustration={affective_state_pred[3]:.2f}'
            basic_emotion_text = f'Neutral={basic_emotion_pred[0]:.2f}, Happiness={basic_emotion_pred[1]:.2f}, Surprise={basic_emotion_pred[2]:.2f}, Sadness={basic_emotion_pred[3]:.2f}, Anger={basic_emotion_pred[4]:.2f}'

            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
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


            response_values[0] = str(np.round(x, 2))
            response_values[1] = str(np.round(y, 2))
            response_values[2] = eye_gaze_horizontal
            response_values[3] = eye_gaze_vertical
            response_values[4] = eye_gaze_text
            response_values[5] = affective_state_text
            response_values[6] = basic_emotion_text

            break


def get_predictions(video_feed):
    detect_engagement_factors(video_feed)


def gen(video_feed):
    while True:
        frame = output_frame(video_feed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#mapping the URL to the function which will send the video feed as frames
@app.route('/')
@cross_origin(supports_credentials=True)
def video_feed():
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
@cross_origin(supports_credentials=True)
def my_profile():
    get_predictions(video)
    response_body = {
        "face_angle_horizontal": response_values[0],
        "face_angle_vertical": response_values[1],
        "eye_gaze_horizontal": response_values[2],
        "eye_gaze_vertical": response_values[3],
        "eye_gaze_text": response_values[4],
        "affective_state_text": response_values[5],
        "basic_emotion_text": response_values[6],
    }
    # response = make_response(jsonify(response_body))
    # response.headers['Access-Control-Allow-Credentials'] = 'true'
    # Specify the domain of your frontend app here. Use '*' with caution.
    # response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'

    return response_body

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)

