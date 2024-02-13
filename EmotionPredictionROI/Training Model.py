import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers.legacy import Adam
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import os

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


def graphs(history):
    #Accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    #Loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

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


def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(48, 192, 3)))
    # First conv block
    model.add(kl.Conv2D(filters=128, kernel_size=3, padding='same', strides=2))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Second conv block
    model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Third conv block
    model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(1024))
    # Second Fc
    model.add(kl.Dense(256))
    # Output FC with sigmoid at the end
    model.add(kl.Dense(4, activation='sigmoid', name='prediction'))
    return model

def simple_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 192, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer with sigmoid activation for multi-label classification
    model.add(layers.Dense(4, activation='sigmoid'))

    return model


def vgg16_model():
    # Create VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 192, 3))

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

def model_from_scratch(input_shape=(48, 192, 3)):
    num_features = 64

    model = Sequential()

    # Stage 1
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape))
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
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same'))
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

    model.add(Dense(4, activation='sigmoid'))

    return model


def saveModelWeights(model, test_acc):
    # Serialize and save model to JSON
    model_name = f'model_{test_acc:.4f}'
    model_json = model.to_json()
    with open(f'{model_name}.json', 'w') as json_file:
        json_file.write(model_json)
    # Save weights to JSON
    model.save_weights(f'{model_name}.h5')


# Construct paths for each dataset
train_df['ClipID'] = train_df.apply(lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Train/bnw_faces'), axis=1)
test_df['ClipID'] = test_df.apply(lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Test/bnw_faces'), axis=1)
validation_df['ClipID'] = validation_df.apply(
    lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Validation/bnw_faces'), axis=1)

# Flatten the DataFrame to have one row per frame
train_df = train_df.explode('ClipID').reset_index(drop=True)
test_df = test_df.explode('ClipID').reset_index(drop=True)
validation_df = validation_df.explode('ClipID').reset_index(drop=True)

# train_df_normalized = normalize_class_values(train_df)
# test_df_normalized = normalize_class_values(test_df)
# validation_df_normalized = normalize_class_values(validation_df)

classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(24, 96),
    batch_size=64,
    class_mode='raw',
    shuffle=True,
    preprocessing_function=augment_image
)

images, labels = next(train_generator)
plot_images(images, labels, 'Train Images')


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(24, 96),
    batch_size=64,
    class_mode='raw',
    shuffle=True,
)

test_images, labels = next(test_generator)
plot_images(test_images, labels, 'Test Images')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(24, 96),
    batch_size=64,
    class_mode='raw',
    shuffle=True,
)

val_images, labels = next(validation_generator)
plot_images(val_images, labels,'Validation Images')

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

# model = model_from_scratch()
#
#
# model.summary()
# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
#
# # Train the model
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=validation_generator,
#     callbacks=callbacks
# )
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])  # Adjust limits for better visualization
# plt.legend(loc='lower right')
# plt.show()
#
# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')

model = network()
model.summary()
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])

# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
    print(f"Training fold {fold + 1}")

    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_fold,
        x_col='ClipID',
        y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
        target_size=(48, 192),
        batch_size=64,
        class_mode='raw',
        shuffle=True,
        preprocessing_function=augment_image
    )

    images, labels = next(train_generator)
    plot_images(images, labels, 'Train Images')

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='ClipID',
        y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
        target_size=(48, 192),
        batch_size=64,
        class_mode='raw',
        shuffle=True,
    )

    images, labels = next(test_generator)
    plot_images(images, labels, 'Test Images')

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=val_fold,
        x_col='ClipID',
        y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
        target_size=(48, 192),
        batch_size=64,
        class_mode='raw',
        shuffle=True,
    )

    images, labels = next(validation_generator)
    plot_images(images, labels, 'Validation Images')

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

    # Train the model
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Evaluate the model on the test set
    test_images, test_labels = next(test_generator)
    test_predictions = model.predict(test_images)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(f'Test accuracy: {test_acc}')

    graphs(history)

    saveModelWeights(model, test_acc)

