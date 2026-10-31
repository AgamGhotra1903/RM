"""
configs/config.py
=================
Master configuration for ERA5 Broad-UNet temperature forecasting.
Defines dataset paths, spatial grid, temporal parameters, splits, and model hyperparameters.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("~/ERA5_Radiation/final").expanduser().resolve()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Meteorological Features
INPUT_VARIABLES = ["t2m", "d2m", "msl", "sp", "slhf", "ssr"]
TARGET_VARIABLE = "t2m"
NUM_INPUT_FEATURES = len(INPUT_VARIABLES)  # 6
TARGET_INDEX = INPUT_VARIABLES.index(TARGET_VARIABLE)  # 0

# Spatial Grid
ORIGINAL_LAT = 28
ORIGINAL_LON = 34
ORIGINAL_SHAPE = (ORIGINAL_LAT, ORIGINAL_LON)

PADDED_LAT = 32
PADDED_LON = 48
PADDED_SHAPE = (PADDED_LAT, PADDED_LON)

# Symmetric padding amounts: (before, after)
PAD_LAT = (2, 2)  # 2 + 28 + 2 = 32
PAD_LON = (7, 7)  # 7 + 34 + 7 = 48

# Temporal Parameters
INTERVAL_HOURS = 2
LAGS = 6  # 6 observations = 12 hours lookback
HORIZONS = [1, 2, 4, 8, 12, 24, 36]  # Forecast steps: +2h, +4h, +8h, +16h, +24h, +48h, +72h
HORIZON_TO_HOURS = {1: 2, 2: 4, 4: 8, 8: 16, 12: 24, 24: 48, 36: 72}
MAX_HORIZON = max(HORIZONS)  # 36 steps (72 hours)

# Strict Chronological Splits (100% Disjoint, Zero Leakage)
# Train: 2011 to 2021 (11 full years, 48,216 timestamps)
# Val:   2022 to 2023 (2 full years, 8,760 timestamps)
# Test:  2024 to 2025 (2 full years, 8,772 timestamps)
TRAIN_YEARS = list(range(2011, 2022))
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024, 2025]

# Model & Optimization Hyperparameters
BASE_FILTERS = 16
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
RANDOM_SEED = 42

# Activation & Loss
ACTIVATION = "linear"  # Linear activation for continuous temperature regression
NORM_METHOD = "zscore"  # 'zscore' or 'minmax' strictly derived from training partition

