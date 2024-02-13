import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# Load the CSV files containing image labels
train_df = pd.read_csv('../Dataset/Labels/TrainLabels.csv')
test_df = pd.read_csv('../Dataset/Labels/TestLabels.csv')
validation_df = pd.read_csv('../Dataset/Labels/ValidationLabels.csv')

# Specify K for KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True)

# Define image data generators for training, testing, and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=20,
    # horizontal_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

max_possible_value = 3



def extract_roi(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for Haar cascades
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascades for face, eyes, and mouth
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Return the whole face region and eyes and mouth
        x, y, w, h = faces[0]
        face_roi = cv2.resize(image[y:y + h, x:x + w], (112, 112))  # Resize to match your model's input size

        # Extract eyes within the face region
        face_gray = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        if len(eyes) >= 2:
            eye1_roi = cv2.resize(face_gray[eyes[0][1]:eyes[0][1] + eyes[0][3], eyes[0][0]:eyes[0][0] + eyes[0][2]], (56, 28))
            eye2_roi = cv2.resize(face_gray[eyes[1][1]:eyes[1][1] + eyes[1][3], eyes[1][0]:eyes[1][0] + eyes[1][2]], (56, 28))
        else:
            eye1_roi = eye2_roi = None

        # Extract mouth within the face region
        mouths = mouth_cascade.detectMultiScale(face_gray)
        if len(mouths) > 0:
            mouth_roi = cv2.resize(face_gray[mouths[0][1]:mouths[0][1] + mouths[0][3], mouths[0][0]:mouths[0][0] + mouths[0][2]], (56, 28))
        else:
            mouth_roi = None

        return eye1_roi, eye2_roi, mouth_roi

    # Return None if no face is found
    return None, None, None, None


def load_and_preprocess_image(image_path):
    eye1_roi, eye2_roi, mouth_roi = extract_roi(image_path)

    # Check if any ROI is None
    if any(x is None for x in (eye1_roi, eye2_roi, mouth_roi)):
        return None

    # Combine individual ROIs into a single image
    combined_image = np.hstack((eye1_roi, eye2_roi, mouth_roi))

    # Resize the combined image to the desired size (56, 28)
    combined_image = cv2.resize(combined_image, (56, 28))

    return augment_image(combined_image)

def augment_image(image):
    # Mirror flip
    # black_white = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flipped = tf.image.flip_left_right(image)
    # Transpose flip
    transposed = tf.image.transpose(image)
    # Saturation
    saturated = tf.image.adjust_saturation(image, 3)
    # Brightness
    brightness = tf.image.adjust_brightness(image, 0.4)
    # Contrast
    contrast = tf.image.random_contrast(image, lower=0.0, upper=1.0)

    # Resize at the end
    images = [flipped, transposed, saturated, brightness, contrast]
    return images

def plot_images(images, labels, title):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 9)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def normalize_class_values(df):
    classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    max_possible_value = 3  # Adjust this based on the actual range of your labels

    for class_name in classes:
        df[class_name] = df[class_name] / max_possible_value

    return df


# Update the 'ClipID' column in the CSV files to include the full path to each frame
def generate_full_paths(row, dataset_dir):
    video_name = row['ClipID'].strip()  # Remove any leading or trailing whitespaces
    full_paths = [os.path.join(dataset_dir, f'{video_name[:-4]}{i}_face.jpg') for i in range(1, 10)]
    return full_paths


def model_from_scratch():
    num_features = 64

    model = Sequential()

    # Stage 1
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=(112, 112, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))

    # Stage 2
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Stage 3
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # Stage 4
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Stage 5
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(4, activation='softmax'))

    return model

def vgg16_model():
    # Create VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(58, 26, 3))

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='sigmoid'))

    # Freeze the layers of the VGG16 model
    for layer in base_model.layers:
        layer.trainable = False

    return model

def   saveModelWeights(model,test_acc):
    # Serialize and save model to JSON
    model_name = f'model_{test_acc:.4f}'
    model_json = model.to_json()
    with open(f'{model_name}.json', 'w') as json_file:
        json_file.write(model_json)
    # Save weights to JSON
    model.save_weights(f'{model_name}.h5')


# Construct paths for each dataset
train_df['ClipID'] = train_df.apply(lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Train/cropped_faces'), axis=1)
test_df['ClipID'] = test_df.apply(lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Test/cropped_faces'), axis=1)
validation_df['ClipID'] = validation_df.apply(
    lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Validation/cropped_faces'), axis=1)

# Flatten the DataFrame to have one row per frame
train_df = train_df.explode('ClipID').reset_index(drop=True)
test_df = test_df.explode('ClipID').reset_index(drop=True)
validation_df = validation_df.explode('ClipID').reset_index(drop=True)

train_df_normalized = normalize_class_values(train_df)
test_df_normalized = normalize_class_values(test_df)
validation_df_normalized = normalize_class_values(validation_df)

classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True
)

lrScheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    earlyStopping,
    lrScheduler,
]

# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df_normalized)):
    print(f"Training fold {fold + 1}")

    train_fold = train_df_normalized.iloc[train_idx]
    val_fold = train_df_normalized.iloc[val_idx]

    train_images = []
    train_labels = []
    for idx, row in train_fold.iterrows():
        clip_id = row['ClipID']
        label_values = [row[class_name] for class_name in classes]
        for i in range(1, 10):
            image_path = f'{clip_id[:-4]}.jpg'
            roi = load_and_preprocess_image(image_path)  # Adjust roi_type as needed
            if roi is not None:
                train_images.append(roi)
                train_labels.append(label_values)

    print("image append finished")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Evaluate the model on the test set
    val_images = []
    val_labels = []
    for idx, row in val_fold.iterrows():
        clip_id = row['ClipID']
        label_values = [row[class_name] for class_name in classes]
        for i in range(1, 10):
            image_path = f'{clip_id[:-4]}.jpg'
            roi = load_and_preprocess_image(image_path)  # Adjust roi_type as needed
            if roi is not None:
                val_images.append(roi)
                val_labels.append(label_values)

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    model = vgg16_model()
    model.summary()
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_images, train_labels,
        epochs=50,
        steps_per_epoch=len(train_images) // 64,
        validation_data=(val_images, val_labels),
        callbacks=callbacks
    )

    # Evaluate the model on the test set
    test_images = []
    test_labels = []
    for idx, row in test_df_normalized.iterrows():
        clip_id = row['ClipID']
        label_values = [row[class_name] for class_name in classes]
        for i in range(1, 10):
            image_path = f'{clip_id[:-4]}.jpg'
            roi = load_and_preprocess_image(image_path)  # Adjust roi_type as needed
            if roi is not None:
                test_images.append(roi)
                test_labels.append(label_values)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy for fold {fold + 1}: {test_acc}')

    saveModelWeights(model,test_acc)
