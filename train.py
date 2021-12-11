import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.debugging.set_log_device_placement(True)

NUM_CLASSES = 0


def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, x)


def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def parse_data(x, y):
    image = read_image(x.decode(), config.IMG_TRIM_SIZE)
    label = [0] * NUM_CLASSES
    label[y] = 1
    label = np.array(label).astype(np.int32)

    return image, label


def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((config.IMG_TRIM_SIZE, config.IMG_TRIM_SIZE, 3))
    y.set_shape(NUM_CLASSES)
    return x, y


def tf_dataset(x, y, batch=8):
    return tf.data.Dataset.from_tensor_slices((x, y))\
        .map(tf_parse)\
        .batch(batch)\
        .repeat()


if __name__ == "__main__":
    labels_df = pd.read_csv(config.LABELS_FILE_PATH)

    breed = labels_df["breed"].unique()
    NUM_CLASSES = len(breed)

    print("Number of Breed: ", NUM_CLASSES)

    breed2id = {name: i for i, name in enumerate(breed)}

    ids = []
    labels = []

    for filename in os.listdir(config.TRAIN_PATH):
        if filename.endswith(config.IMG_FILE_EXTENSION):
            img_id = filename.removesuffix(config.IMG_FILE_EXTENSION)
            labels.append(breed2id[list(labels_df[labels_df.id == img_id]["breed"])[0]])
            ids.append(os.path.join(config.TRAIN_PATH, filename))
            continue
        else:
            continue

    # Splitting the dataset
    train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    # Parameters
    lr = 1e-4
    batch = 16

    # Model
    model = build_model(config.IMG_TRIM_SIZE, NUM_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])
    print(f"summary: {model.summary()}")

    # Dataset
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    # Training
    callbacks = [
        ModelCheckpoint(config.MODEL_NAME, verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
    ]
    train_steps = (len(train_x) // batch) + 1
    valid_steps = (len(valid_x) // batch) + 1
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              validation_data=valid_dataset,
              epochs=config.NUM_EPOCHS,
              callbacks=callbacks)
