import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from lime import lime_image

# Load the trained model architecture from JSON file
with open('Saved-Models/model_0.8750.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the trained model weights
loaded_model.load_weights('Saved-Models/model_0.8750.h5')

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


input_image_path = '../Dataset/Image_Dataset_3/Test/face_mesh_rois_vertical/50004410095.jpg'

img_original = cv2.imread(input_image_path)
img_original_1 = cv2.resize(img_original, (144, 80))

# Normalize the image pixel values to the range [0, 1]
img_original = img_original_1 / 255.0
# Expand the dimensions of the image to match the input shape expected by the model
img_original = np.expand_dims(img_original, axis=0)

prediction = loaded_model.predict(img_original)[0][3]  # Access the first element for ASD probability
# print("prediction: ", prediction)
# print("prediction: {:.2f}".format(prediction))

# Visualize the Grad-CAM heatmap
heatmap = generate_grad_cam(loaded_model, img_original, 3)

# Resize heatmap to match the size of the original image
heatmap = cv2.resize(heatmap, (144, 80))

# Apply colormap for better visualization
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

heatmap = heatmap.astype(img_original_1.dtype)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(img_original_1, 0.6, heatmap, 0.4, 0)

# Display the superimposed image
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.text(1, -5, "class: Boredom prediction: {:.2f}".format(prediction), color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.axis('off')
plt.show()

# LIME explainations

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_original_1, loaded_model.predict, top_labels=1, hide_color=0, num_samples=1000)


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
mask = cv2.resize(mask, (144, 80), interpolation=cv2.INTER_NEAREST)
img_original_1[mask > 0.5] = (0, 255, 0)
plt.imshow(cv2.cvtColor(img_original_1, cv2.COLOR_BGR2RGB))
# plt.imshow(mask, cmap='jet', alpha=0.6)
plt.axis('off')
plt.show()