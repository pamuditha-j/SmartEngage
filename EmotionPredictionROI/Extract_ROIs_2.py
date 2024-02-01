import cv2
import dlib
import os
import numpy as np


def extract_roi(image_path, predictor_path):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces
    faces = detector(image)

    if len(faces) > 0:
        # Use the first detected face
        face = faces[0]

        # Get the facial landmarks
        landmarks = predictor(image, face)

        # Extract eyes and mouth based on facial landmarks
        eye1_roi = image[landmarks.part(36).y:landmarks.part(39).y, landmarks.part(36).x:landmarks.part(39).x]
        eye2_roi = image[landmarks.part(42).y:landmarks.part(45).y, landmarks.part(42).x:landmarks.part(45).x]
        mouth_roi = image[landmarks.part(48).y:landmarks.part(54).y, landmarks.part(48).x:landmarks.part(54).x]

        # Draw rectangles around the ROIs for visualization
        cv2.rectangle(image, (landmarks.part(36).x, landmarks.part(36).y),
                      (landmarks.part(39).x, landmarks.part(39).y), (0, 255, 0), 2)  # Eye1
        cv2.rectangle(image, (landmarks.part(42).x, landmarks.part(42).y),
                      (landmarks.part(45).x, landmarks.part(45).y), (0, 255, 0), 2)  # Eye2
        cv2.rectangle(image, (landmarks.part(48).x, landmarks.part(48).y),
                      (landmarks.part(54).x, landmarks.part(54).y), (0, 255, 0), 2)  # Mouth

        cv2.imshow("Processed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Resize the ROIs
        if not eye1_roi.size or not eye2_roi.size or not mouth_roi.size:
            return None, None, None

        # Resize the ROIs
        eye1_roi = cv2.resize(eye1_roi, (56, 112))
        eye2_roi = cv2.resize(eye2_roi, (56, 112))
        mouth_roi = cv2.resize(mouth_roi, (56, 112))

        return eye1_roi, eye2_roi, mouth_roi

    # Return None if no face is found
    return None, None, None


def combine_and_save_image(input_folder, output_folder, predictor_path):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(root, file)

            # Extract ROIs
            eye1_roi, eye2_roi, mouth_roi = extract_roi(image_path, predictor_path)

            # Check if any ROI is None
            if any(x is None for x in (eye1_roi, eye2_roi, mouth_roi)):
                continue

            # Combine individual ROIs into a single image
            combined_image = np.hstack((eye1_roi, eye2_roi, mouth_roi))

            # Resize the combined image to the desired size (56, 28)
            combined_image = cv2.resize(combined_image, (168, 112))

            # Save the combined image in the output folder
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, combined_image)


if __name__ == "__main__":
    input_folder1 = "../Dataset/Image_Dataset_2/Train/cropped_faces"
    input_folder2 = "../Dataset/Image_Dataset_2/Validation/cropped_faces"
    input_folder3 = "../Dataset/Image_Dataset_2/Test/cropped_faces"
    output_folder1 = "../Dataset/Image_Dataset_2/Train/rois"
    output_folder2 = "../Dataset/Image_Dataset_2/Validation/rois"
    output_folder3 = "../Dataset/Image_Dataset_2/Test/rois"

    predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Update with the correct path

    # Ensure the output folder exists
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    os.makedirs(output_folder3, exist_ok=True)

    # Process images from three input folders
    combine_and_save_image(input_folder1, output_folder1, predictor_path)
    combine_and_save_image(input_folder2, output_folder2, predictor_path)
    combine_and_save_image(input_folder3, output_folder3, predictor_path)
