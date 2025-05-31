# config.py

# Data paths
CASCADE_PATH = './data/cascade/haarcascade_frontalface_default.xml'
FONT_PATH = './data/font/Calibri Regular.ttf' # Adjusted path for root scripts
MODEL_PATH = './data/model/facenet_keras.h5'
EMBEDDINGS_PATH = './data/arrays/embeddings.npz'
VARS_PATH = './data/arrays/vars.npz'

# Image paths for data generation and testing
IMAGES_DIR = './data/images/'
FACES_DIR = './data/faces/'
TEST_IMAGES_DIR = './test/'
PREDICTED_IMAGES_DIR = './test/predicted/'

# Note: Paths used within scripts in the 'script' directory might need adjustment
# if they are relative to the script's location.
# For example, '../data/font/Calibri Regular.ttf' would become './data/font/Calibri Regular.ttf'
# if the script using it is in the root, or accessed via this config file.
# We will adjust script imports and path usage in later steps.
