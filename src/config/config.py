import torch

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
MODEL_VERSION = '0.1.0'
EPOCHS = 10000
BATCH_SIZE = 64
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_ACCURACY = 0.8
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
QUALITY_THRESHOLD = 5
RANDOM_STATE = 42
LEARNING_RATE = 0.001
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
PLAUSIBLE_RANGES = {
    'fixed acidity':        (3.0, 16.0),
    'volatile acidity':     (0.0, 1.2),
    'citric acid':          (0.0, 1.0),
    'residual sugar':       (0.0, 8.0),
    'chlorides':            (0.0, 0.2),
    'free sulfur dioxide':  (1.0, 72.0),
    'total sulfur dioxide': (1.0, 200.0),
    'density':              (0.985, 1.010),
    'pH':                   (2.5, 4.5),
    'sulphates':            (0.0, 1.5),
    'alcohol':              (8.0, 15.5),
}
