import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.layers as kl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV files containing image labels
train_df = pd.read_csv('../Dataset/Labels/TrainLabels.csv')
test_df = pd.read_csv('../Dataset/Labels/TestLabels.csv')
validation_df = pd.read_csv('../Dataset/Labels/ValidationLabels.csv')

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

# def custom_data_generator(generator):
#     for x_batch, y_batch in generator:
#         # Normalize labels here
#         max_possible_value = 3  # Adjust this based on the actual range of your labels
#         y_batch_normalized = y_batch.astype(np.float32) / max_possible_value
#         yield x_batch, y_batch_normalized

def plot_images(images, title):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 9)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def normalize_class_values(df):
    classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    max_possible_value = 3  # Adjust this based on the actual range of your labels

    for class_name in classes:
        df[class_name] = df[class_name] / max_possible_value

    return df

# Update the 'filename' column in the CSV files to include the full path to each frame
# Update the 'ClipID' column in the CSV files to include the full path to each frame
def generate_full_paths(row, dataset_dir):
    video_name = row['ClipID'].strip()  # Remove any leading or trailing whitespaces
    full_paths = [os.path.join(dataset_dir, f'{video_name[:-4]}{i}_face.jpg') for i in range(1, 10)]
    return full_paths


# def data_augmentation(x_train):
#     shift = 0.1
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         horizontal_flip=True,
#         height_shift_range=shift,
#         width_shift_range=shift)
#     datagen.fit(x_train)
#     return datagen


def myModel():
    input_shape = (48, 48, 3)
    classes = 4
    num_features = 64
    model = Sequential()

    # Stage 1
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))
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
    model.add(Dropout(0.5))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))

    # Stage 4
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Stage 5
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='sigmoid'))

    return model


def resnet():
    # Build a ResNet model
    model = models.Sequential()
    model.add(tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(48, 48, 3),
        pooling=None
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='sigmoid'))
    return model

def mobilenet():
    base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')

    model = models.Sequential()
    model.add(base_model)
    # model.add(layers.Conv2D(1, (1, 1), activation='relu'))  # Add a 1x1 convolution to convert 1 channel to 3 channels
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Lambda(lambda x: x / max_possible_value, input_shape=(1,)))
    model.add(layers.Dense(4, activation='sigmoid'))

    return model

def vgg16():
    base_model = VGG16(input_shape=(48, 48, 1), include_top=False, weights='imagenet')

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))

    return model
def xception():
    # Build an Xception model
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(96, 96, 3),  # Set the input shape for grayscale images
        pooling=None
    )

    # Add additional layers to the Xception base model
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))

    return model

def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(48, 48, 3)))

    vgg = VGG16(weights='imagenet', input_shape=(48, 48, 3), include_top=False)
    vgg.trainable = False
    model.add(vgg)

    # mobnet = MobileNetV2(weights='imagenet', input_shape=(48, 48, 3), include_top=False)
    # mobnet.trainable = False
    # model.add(mobnet)

    # First conv block
    # model.add(kl.Conv2D(filters=128, kernel_size=3, padding='same', strides=2))
    # model.add(tf.keras.layers.ReLU())
    # model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # # Second conv block
    # model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
    # model.add(tf.keras.layers.ReLU())
    # model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # # Third conv block
    # model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
    # model.add(tf.keras.layers.ReLU())
    # model.add(kl.MaxPooling2D(pool_size=(2, 2)))


    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(1024))
    model.add(layers.Dropout(0.5))
    # Second Fc
    model.add(kl.Dense(256))
    model.add(layers.Dropout(0.5))
    # Output FC with sigmoid at the end
    model.add(kl.Dense(4, activation='sigmoid'))
    return model

def saveModelWeights(model, test_acc):
    # Serialize and save model to JSON
    model_json = model.to_json()
    with open('Saved-Models\\model.json', 'w') as json_file:
        json_file.write(model_json)
    # Save weights to JSON
    model.save_weights('Saved-Models\\model.h5')


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

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df_normalized,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(48, 48),
    batch_size=64,
    class_mode='raw',
    shuffle=True,
    # color_mode='grayscale'
)

train_images, _ = next(train_generator)
plot_images(train_images, 'Train Images')



# Plot some augmented images
# augmented_images = train_datagen.flow(original_images, shuffle=False, batch_size=64)
# plot_images(augmented_images, 'Augmented Images')



test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df_normalized,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(48, 48),
    batch_size=64,
    class_mode='raw',
    shuffle=True,
    # color_mode='grayscale'
)

test_images, _ = next(test_generator)
plot_images(test_images, 'Test Images')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df_normalized,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(48, 48),
    batch_size=64,
    class_mode='raw',
    shuffle=False,
    # color_mode='grayscale'
)

val_images, _ = next(validation_generator)
plot_images(val_images, 'Validation Images')

# normalized_validation_generator = custom_data_generator(validation_generator)

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

# model = myModel()
# model = resnet()
model = mobilenet()
# model = vgg16()
# model = xception()
# model = network()

model.summary()
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    # steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    callbacks=callbacks
)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])  # Adjust limits for better visualization
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

saveModelWeights(model, test_acc)
