#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import tensorflow.keras.layers as kl
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from DatasetPreprocessing import DataPreprocessing
import os
from tqdm import tqdm

BATCH_SIZE = 64
LR = 0.005
EPOCHS = 100
data_augmentation = True
checkpoint_dir = 'checkpoints/'

os.makedirs(checkpoint_dir, exist_ok=True)


# def networkOne():
#     model = tf.keras.Sequential()
#     model.add(kl.InputLayer(input_shape=(48, 48, 1)))
#
#     # First conv block
#     model.add(kl.Conv2D(filters=128, kernel_size=3, padding='same', strides=2))
#     model.add(tf.keras.layers.ReLU())
#     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
#     # Second conv block
#     model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
#     model.add(tf.keras.layers.ReLU())
#     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
#     # Third conv block
#     model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
#     model.add(tf.keras.layers.ReLU())
#     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
#
#     # Flatten
#     model.add(kl.Flatten())
#     # First FC
#     model.add(kl.Dense(1024))
#     # Second Fc
#     model.add(kl.Dense(256))
#     # Output FC with sigmoid at the end
#     model.add(kl.Dense(4, activation='sigmoid', name='prediction'))
#     return model

def networkTwo():

    num_features = 64

    model = Sequential()

    # Stage 1
    model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=(48, 48, 1)))
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

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Track progress
    train_loss_avg.update_state(loss_value)
    train_accuracy.update_state(macro_f1(y, logits))
    #acc_update(y, logits)
    return loss_value

@tf.function
def test_step(model, x, y, set_name):
    logits = model(x)
    if set_name == 'val':
        val_accuracy.update_state(y, logits)
    else:
        test_accuracy.update_state(y, logits)


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()

    # Open train set
    tfrecord_path = 'tfrecords/train.tfrecords'
    train_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    train_set = train_set.map(preprocessing_class.decode)
    train_set = train_set.shuffle(1)
    train_set = train_set.batch(BATCH_SIZE)

    # Open test set
    tfrecord_path = 'tfrecords/test.tfrecords'
    test_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    test_set = test_set.map(preprocessing_class.decode)
    test_set = test_set.shuffle(1)
    test_set = test_set.batch(BATCH_SIZE)

    # Open val set
    tfrecord_path = 'tfrecords/val.tfrecords'
    val_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    val_set = val_set.map(preprocessing_class.decode)
    val_set = val_set.shuffle(1)
    val_set = val_set.batch(BATCH_SIZE)

    # Create the model
    model = networkTwo()

    # Optimizers and metrics
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)

    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Mean()

    last_models = sorted(os.listdir(checkpoint_dir))
    if last_models:
        last_model_path = checkpoint_dir + '/' + last_models[-1]
        first_epoch = int(last_models[-1].split("_")[1]) + 1
        print("First epoch is ", first_epoch)
        model = tf.keras.models.load_model(last_model_path)
    else:
        first_epoch = 0
        model = networkTwo()

    model.summary()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    # Train
    for epoch in tqdm(range(first_epoch, EPOCHS+1), total=EPOCHS+1-first_epoch):
        try:
            # Initialize tqdm for progress bar
            progress_bar = tqdm(total=len(list(train_set)), desc=f"Epoch {epoch}/{EPOCHS}", position=0, leave=True)

            # Training loop
            for x_batch_train, y_batch_train in train_set:
                # Do step
                loss_value = train_step(x_batch_train, y_batch_train)

                # Update tqdm progress bar
                progress_bar.update(1)

            # Close the tqdm progress bar for the training loop
            progress_bar.close()

            # Test on validation set
            for x_batch_val, y_batch_val in val_set:
                test_step(model, x_batch_val, y_batch_val, 'val')

            print(f"\nEpoch {epoch}/{EPOCHS} - Train Loss: {train_loss_avg.result()}, Train Accuracy: {train_accuracy.result()}, Val Accuracy: {val_accuracy.result()}")

            # Reset training metrics at the end of each epoch
            train_accuracy.reset_states()
            val_accuracy.reset_states()

            if epoch % 10 == 0:
                tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)),
                                           save_format="h5")

        except KeyboardInterrupt:
            print("Keyboard Interruption...")
            # Save model
            tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)),
                                       save_format="h5")
            break

    epochs = 50
    batch_size = 64

    # Training model from scratch
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [
        earlyStopping,
        lrScheduler,
    ]
    model = networkTwo()
    model.summary()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    trainingHistory = model.fit(train_set, epochs=EPOCHS,
                                steps_per_epoch=len(list(train_set)),
                                validation_data=val_set,
                                callbacks=callbacks)
    test_loss, test_acc = model.evaluate(test_set, batch_size=batch_size)

    # Test on validation set
    for x_batch_test, y_batch_test in test_set:
        test_step(model, x_batch_test, y_batch_test, 'test')
    test_set_acc = test_accuracy.result().numpy()
    print("Accuracy on test set is ", test_set_acc)