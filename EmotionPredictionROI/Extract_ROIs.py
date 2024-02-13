import cv2
import os
import numpy as np
import dlib


# def extract_roi(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#
#     # Load Haar cascades for face, eyes, and mouth
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#     nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
#     mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
#
#     # Detect faces
#     faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     if len(faces) > 0:
#         # Return the whole face region and eyes and mouth
#         x, y, w, h = faces[0]
#         face_roi = cv2.resize(image[y:y + h, x:x + w], (224, 224))  # Resize to match your model's input size
#
#         # Extract eyes within the face region
#         face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
#         eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
#         if len(eyes) == 2:
#             # Increase the size of the bounding box for the eyes
#             eyes_x = max(0, min(eyes[0][0], eyes[1][0]) - 10)
#             eyes_y = max(0, min(eyes[0][1], eyes[1][1]) - 10)
#             eyes_w = min(face_roi.shape[1] - eyes_x,
#                          max(eyes[0][0] + eyes[0][2], eyes[1][0] + eyes[1][2]) - eyes_x + 10)
#             eyes_h = min(face_roi.shape[0] - eyes_y,
#                          max(eyes[0][1] + eyes[0][3], eyes[1][1] + eyes[1][3]) - eyes_y + 10)
#
#             eyes_roi = cv2.resize(face_gray[eyes_y:eyes_y + eyes_h, eyes_x:eyes_x + eyes_w], (112, 112))
#         else:
#             eyes_roi = None
#
#         # Extract nose within the face region
#         noses = nose_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
#         if len(noses) > 0:
#             # Create a bounding box around the nose
#             nose_x = max(0, noses[0][0] - 10)
#             nose_y = max(0, noses[0][1] - 10)
#             nose_w = min(face_roi.shape[1] - nose_x, noses[0][2] + 20)
#             nose_h = min(face_roi.shape[0] - nose_y, noses[0][3] + 20)
#
#             nose_roi = cv2.resize(face_gray[nose_y:nose_y + nose_h, nose_x:nose_x + nose_w], (56, 56))
#         else:
#             nose_roi = None
#
#         # Extract mouth within the face region
#         mouths = mouth_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
#         if len(mouths) > 0:
#             # Create a bounding box that covers both nose and mouth
#             combined_x = min(nose_x, mouths[0][0])
#             combined_y = min(nose_y, mouths[0][1])
#             combined_w = max(nose_x + nose_w, mouths[0][0] + mouths[0][2]) - combined_x
#             combined_h = max(nose_y + nose_h, mouths[0][1] + mouths[0][3]) - combined_y
#
#             mouth_roi = cv2.resize(
#                 face_gray[combined_y:combined_y + combined_h, combined_x:combined_x + combined_w], (112, 112))
#         else:
#             mouth_roi = None
#
#         return eyes_roi, mouth_roi
#
#     # Return None if no face is found
#     return None, None

def extract_roi(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Load Dlib shape predictor for 68-point face landmarks
    predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Provide the correct path to your shape predictor file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces
    faces = detector(image)

    if len(faces) > 0:
        # Return the whole face region and eyes and mouth
        shape = predictor(image, faces[0])
        landmarks = shape.parts()

        face_rect = faces[0]

        # Return the whole face region
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        face_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if w > 0 and h > 0:
            x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1] - x), min(h, image.shape[0] - y)

            face_roi = cv2.resize(face_gray[y:y + h, x:x + w], (48, 48))

            # Extract eyes within the face region
            face_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extracting the region around eyes based on landmark points (adjust the points based on your needs)
            eye_points = landmarks[36:48]  # Assuming landmarks 36-47 correspond to the eyes
            eyes_x = min(point.x for point in eye_points) - 25
            eyes_y = min(point.y for point in eye_points) - 20
            eyes_w = max(point.x for point in eye_points) - eyes_x + 45
            eyes_h = max(point.y for point in eye_points) - eyes_y + 10

            if eyes_y >= 0 and eyes_x >= 0 and (eyes_y + eyes_h) <= face_gray.shape[0] and (eyes_x + eyes_w) <= face_gray.shape[1]:
                eyes_roi = cv2.resize(face_gray[eyes_y:eyes_y + eyes_h, eyes_x:eyes_x + eyes_w], (48, 24))
            else:
                return None, None, None

            # eyes_roi = cv2.resize(face_gray[eyes_y:eyes_y + eyes_h, eyes_x:eyes_x + eyes_w], (48, 24))

            # Extract mouth within the face region
            # Extracting the region around the mouth based on landmark points (adjust the points based on your needs)
            mouth_points = landmarks[48:68]  # Assuming landmarks 48-67 correspond to the mouth
            mouth_x = min(point.x for point in mouth_points) - 25
            mouth_y = min(point.y for point in mouth_points) - 15
            mouth_w = max(point.x for point in mouth_points) - mouth_x + 45
            mouth_h = max(point.y for point in mouth_points) - mouth_y + 15

            if mouth_y >= 0 and mouth_x >= 0 and (mouth_y + mouth_h) <= face_gray.shape[0] and (mouth_x + mouth_w) <= face_gray.shape[1]:
                mouth_roi = cv2.resize(face_gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w], (48, 24))
            else:
                return None, None, None

            return face_roi, eyes_roi, mouth_roi

    # Return None if no face is found
    return None, None, None


def combine_and_save_image(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(root, file)

            # Extract ROIs
            face, eyes_roi, mouth_roi = extract_roi(image_path)

            # Check if any ROI is None
            if any(x is None for x in (eyes_roi, mouth_roi)):
                continue

            # Combine individual ROIs into a single image
            combined_image = np.hstack((eyes_roi, mouth_roi))

            # Resize the combined image to the desired size (56, 28)
            combined_image = cv2.resize(combined_image, (96, 24))


            # Save the combined image in the output folder
            output_path_roi = os.path.join(output_folder, 'rois', file)
            output_path_face = os.path.join(output_folder, 'bnw_faces', file)

            cv2.imwrite(output_path_roi, combined_image)
            cv2.imwrite(output_path_face, face)


if __name__ == "__main__":
    input_folder1 = "../Dataset/Image_Dataset_2/Train/cropped_faces"
    input_folder2 = "../Dataset/Image_Dataset_2/Validation/cropped_faces"
    input_folder3 = "../Dataset/Image_Dataset_2/Test/cropped_faces"
    output_folder1 = "../Dataset/Image_Dataset_2/Train"
    output_folder2 = "../Dataset/Image_Dataset_2/Validation"
    output_folder3 = "../Dataset/Image_Dataset_2/Test"

    # Process images from three input folders
    combine_and_save_image(input_folder2, output_folder2)
    combine_and_save_image(input_folder3, output_folder3)
    combine_and_save_image(input_folder1, output_folder1)
