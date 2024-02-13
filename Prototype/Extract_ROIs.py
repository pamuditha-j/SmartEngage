import cv2
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def detect_face(image_path, face_cascade):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haarcascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assume there's only one face, take the first one
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        return face_region, (x, y, w, h)
    else:
        return None, None


def extract_roi(image, face_mesh):
    try:
        # Convert the image to RGB format (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe Face Mesh
        for i in range(30):
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                eyes_roi = None
                mouth_roi = None
                face_roi = None
                # Extract the first face landmarks (assuming only one face per image)
                face_landmarks = results.multi_face_landmarks[0]

                # Extract landmark points
                landmark_points = [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in face_landmarks.landmark]

                # Convert the landmark points to NumPy array
                landmark_points = np.array(landmark_points)

                # Extract eyes region
                eyes_roi = image[int(landmark_points[109, 1]):int(landmark_points[101, 1]),
                           int(landmark_points[162, 0]):int(landmark_points[389, 0])]

                mouth_roi = image[int(landmark_points[36, 1]):int(landmark_points[201, 1]),
                            int(landmark_points[207, 0]):int(landmark_points[427, 0])]

                face_roi = image[int(landmark_points[10, 1]):int(landmark_points[152, 1]),
                            int(landmark_points[234, 0]):int(landmark_points[356, 0])]


                # Resize eyes and mouth regions to a fixed size (adjust as needed)
                eyes_roi = cv2.resize(eyes_roi, (144, 72))
                mouth_roi = cv2.resize(mouth_roi, (144, 72))
                face_roi = cv2.resize(face_roi, (244, 244))
                # Return the extracted ROIs
                return face_roi, eyes_roi, mouth_roi, False
        return None, None, None, True
    except Exception as e:
        print("Error resizing ROI:", e)
        return None, None, None, True


def combine_and_save_image(input_folder, output_folder, face_mesh, face_cascade):
    count = 0
    error_count = 0
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            try:
                count = count + 1
                image_path = os.path.join(root, file)

                # Detect face using Haarcascade
                for i in range(30):
                    face_region, face_rect = detect_face(image_path, face_cascade)
                    if face_region is not None and face_rect is not None:
                        break

                # Extract ROIs using MediaPipe Face Mesh
                face_roi, eyes_roi, mouth_roi, isFailed = extract_roi(face_region, face_mesh)

                if isFailed:
                    error_count = error_count + 1

                # Combine individual ROIs into a single image
                combined_image = np.hstack((eyes_roi, mouth_roi))

                # Resize the combined image to the desired size (adjust as needed)
                combined_image = cv2.resize(combined_image, (288, 72))

                # Plot the combined image
                # plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
                # plt.title('Combined Image')
                # plt.axis('off')
                # plt.show()
            except Exception as e:
                print("Error ", e)
                error_count = error_count + 1
                continue

            # Save the combined image in the output folder
            output_path_roi = os.path.join(output_folder, 'face_mesh_rois', file)
            output_path_face = os.path.join(output_folder, 'face_mesh_face', file)
            print(" Image Count " + str(count) + "  //  Error Count " + str(error_count))
            cv2.imwrite(output_path_roi, combined_image)
            cv2.imwrite(output_path_face, face_roi)
    print(str(input_folder) + "Image Count " + str(count) + "  //  Error Count " + str(error_count))

if __name__ == "__main__":
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize Haarcascade face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Define input and output folders
    input_folder1 = "../Dataset/Image_Dataset_2/Train"
    input_folder2 = "../Dataset/Image_Dataset_2/Validation"
    input_folder3 = "../Dataset/Image_Dataset_2/Test"
    output_folder1 = "../Dataset/Image_Dataset_3/Train"
    output_folder2 = "../Dataset/Image_Dataset_3/Validation"
    output_folder3 = "../Dataset/Image_Dataset_3/Test"

    # Process images from three input folders
    combine_and_save_image(input_folder1, output_folder1, face_mesh, face_cascade)
    combine_and_save_image(input_folder2, output_folder2, face_mesh, face_cascade)
    combine_and_save_image(input_folder3, output_folder3, face_mesh, face_cascade)

    # Release resources
    face_mesh.close()
