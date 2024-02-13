import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import os


def normalize_class_values(df):
    classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    max_possible_value = 3  # Adjust this based on the actual range of your labels

    def g(row):
        for class_name in classes:
            # Normalize and conditionally replace for each value in the row
            row[class_name] = 1 if row[class_name] / max_possible_value > 0.5 else 0
        return row

    # Apply the function to each row using apply or lambda
    df = df.apply(g, axis=1)  # Using 'axis=1' applies to each row
    # Or:
    # df = df.transform(lambda row: g(row), axis=1)

    return df


def generate_full_paths(row, dataset_dir):
    video_name = row['ClipID'].strip()  # Remove any leading or trailing whitespaces
    full_paths = [os.path.join(dataset_dir, f'{video_name[:-4]}{i}_face.jpg') for i in range(1, 10)]
    return full_paths


test_df = pd.read_csv('../Dataset/Labels/TestLabels.csv')
test_datagen = ImageDataGenerator(
    rescale=1./255,
)
test_df['ClipID'] = test_df.apply(lambda row: generate_full_paths(row, '../Dataset/Image_Dataset_2/Test/rois'), axis=1)
test_df = test_df.explode('ClipID').reset_index(drop=True)
# test_df_normalized = normalize_class_values(test_df)


test_df_normalized = normalize_class_values(test_df)

#
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df_normalized,
    x_col='ClipID',
    y_col=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    target_size=(96, 96),
    batch_size=64,
    class_mode='raw',
    shuffle=True
)

# Load model from JSON file
json_file = open('Saved-Models/model_0.8281.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#Load weights and them to model
model.load_weights('Saved-Models/model_0.8281.h5')


model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have trained your model and obtained predictions
predictions = model.predict(test_generator)

# Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Get true labels from the generator
true_labels = np.concatenate([test_generator[i][1] for i in range(len(test_generator))])

precision = precision_score(true_labels, binary_predictions, average=None)
recall = recall_score(true_labels, binary_predictions, average=None)
f1 = f1_score(true_labels, binary_predictions, average=None)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

test_images, test_labels = next(test_generator)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_acc}')

# Reshape true labels to match the shape of binary_predictions
true_labels_reshaped = true_labels.reshape(binary_predictions.shape)

# Compute the confusion matrix
conf_matrix = multilabel_confusion_matrix(true_labels_reshaped, binary_predictions)

print("Confusion Matrix:")
print(conf_matrix)

conf_matrix_2d = conf_matrix.reshape((conf_matrix.shape[0], -1))

fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(conf_mat=conf_matrix_2d,
                     class_names=['Boredom', 'Engagement', 'Confusion', 'Frustration'],
                     colorbar=True,
                     show_normed=True,
                     show_absolute=False)  # Pass the axis object
plt.show()

print(classification_report(true_labels_reshaped, binary_predictions, target_names=['Boredom', 'Engagement', 'Confusion', 'Frustration']))