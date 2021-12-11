import math
import os
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from alive_progress import alive_it
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_image(path, size):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (size, size))
    return (img / 255.0).astype(np.float32)


if __name__ == "__main__":
    labels_df = pd.read_csv(config.LABELS_FILE_PATH)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))

    id2breed = {i: name for i, name in enumerate(breed)}

    model = tf.keras.models.load_model(config.MODEL_NAME)

    for filename in alive_it(os.listdir(config.TEST_PATH), theme='smooth'):
        if filename.endswith(config.IMG_FILE_EXTENSION):
            img_id = filename.removesuffix(config.IMG_FILE_EXTENSION)
            img_path = os.path.join(config.TEST_PATH, filename)

            image = read_image(img_path, config.IMG_TRIM_SIZE)
            image = np.expand_dims(image, axis=0)
            pred = model.predict(image)[0]
            label_idx = np.argmax(pred)
            breed_name = id2breed[label_idx]

            ori_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width, _ = ori_image.shape

            fontScale = min(width, height) / (25 / config.FONT_SIZE)

            fontThickness = math.ceil(width / 200)
            ori_image = cv2.putText(
                img=ori_image,
                text=breed_name,
                org=(5, int(height - (25 * fontScale) + 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontScale,
                color=(255, 0, 0),
                thickness=fontThickness
            )

            print(f"done: {filename}")
            cv2.imwrite(os.path.join(config.RESULTS_PATH, f"{img_id}{config.IMG_FILE_EXTENSION}"), ori_image)

            time.sleep(0.1)

            continue
        else:
            continue
