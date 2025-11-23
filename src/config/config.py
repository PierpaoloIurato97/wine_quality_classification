import torch

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
MODEL_VERSION = '0.1.0'
EPOCHS = 10000
BATCH_SIZE = 128
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_DELTA = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
LABEL = 'label'
