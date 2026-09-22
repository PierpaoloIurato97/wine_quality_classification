DATA_DIR = "data"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
MODEL_VERSION = "0.1.0"
TRAIN_RATIO = 0.6
TEST_RATIO = 0.4
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
LABEL = "label"
QUALITY_THRESHOLD = 5
RANDOM_STATE = 42
N_CLUSTERS = 5
OPTUNA_N_ITER = 100
OPTUNA_C_BOUNDS = (1e-1, 1e1)
OPTUNA_GAMMA_BOUNDS = (1e-1, 1e1)
PLAUSIBLE_RANGES = {
    "fixed acidity": (3.0, 16.0),
    "volatile acidity": (0.0, 1.2),
    "citric acid": (0.0, 1.0),
    "residual sugar": (0.0, 8.0),
    "chlorides": (0.0, 0.2),
    "free sulfur dioxide": (1.0, 72.0),
    "total sulfur dioxide": (1.0, 200.0),
    "density": (0.985, 1.010),
    "pH": (2.5, 4.5),
    "sulphates": (0.0, 1.5),
    "alcohol": (8.0, 15.5),
}
