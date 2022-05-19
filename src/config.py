"""
Before training and evaluation - make sure to select desired/correct settings
"""

# Input dataset related settings
DATA_PATH = "../data/"
INPUT_HEIGHT = 80
INPUT_WIDTH = 2000
INPUT_SHAPE = (INPUT_WIDTH, INPUT_HEIGHT)

TACO_AUGMENTAION_FRACTION = 0.9

# If Long lines augmentation is needed (see paper)
LONG_LINES = True
LONG_LINES_FRACTION = 0.3

# Model training parameters
BATCH_SIZE = 32
EPOCHS = 1000
VOCAB_SIZE = 80
DROPOUT = True
OUTPUT_SHAPE = 500

# Initializing weights from pre-trained 
LOAD = True
LOAD_CHECKPOINT_PATH = "../weights/saved_checkpoint.hdf5"

# Other learning parametes
LEARNING_RATE = 0.01
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997

# Checkpoints parametes
CHECKPOINT_PATH = '../weights/EASTER2--{epoch:02d}--{loss:.02f}.hdf5'
LOGS_DIR = '../logs'
BEST_MODEL_PATH = "../weights/saved_checkpoint.hdf5"