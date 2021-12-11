import os

IMG_FILE_EXTENSION = '.jpg'
IMG_TRIM_SIZE = 224
SRC_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SRC_PATH, 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')
LABELS_FILE_PATH = os.path.join(DATA_PATH, 'labels.csv')
MODEL_NAME = "model.h5"

NUM_EPOCHS = 10
FONT_SIZE = 0.1

