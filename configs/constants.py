# GEE project name (no secrets)
GEE_PROJECT = "coringa-mangrove"

# Coringa Wildlife Sanctuary bounding box
# Paper: 16°42'–17°0'N, 82°13'–82°23'E
CORINGA_BBOX = [82.13, 16.42, 82.23, 17.0]

# Sentinel-2 date range (Sep–Dec 2022)
S2_START = "2022-09-01"
S2_END = "2022-12-31"

# Landsat-8 date range (Sep–Dec 2017)
L8_START = "2017-09-01"
L8_END = "2017-12-31"

# Band names
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
L8_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]  # Blue, Green, Red, NIR, SWIR1, SWIR2

# Spectral index bands (same names, different formulas per sensor)
INDEX_BANDS = ["NDVI", "NDMI", "MNDWI", "GCVI", "SR"]

# All feature bands
S2_FEATURES = S2_BANDS + INDEX_BANDS  # 11 channels
L8_FEATURES = L8_BANDS + INDEX_BANDS  # 11 channels

# Customized Thresholds (CT) from paper Table 2
# Derived by inspecting pixel values in Coringa mangroves via GEE inspector tool
CT_THRESHOLDS = {
    "NDVI": {"min": 0.4, "max": None},        # >= 0.4
    "NDMI": {"min": -0.34, "max": -0.14},      # -0.34 to -0.14
    "MNDWI": {"min": -0.35, "max": -0.14},     # -0.35 to -0.14
    "GCVI": {"min": 3.7, "max": None},          # >= 3.7
    "SR": {"min": 6.5, "max": 11.0},            # 6.5 to 11
}

# Random Forest hyperparams (paper: 100 trees, 5 variables)
RF_N_ESTIMATORS = 100
RF_MAX_FEATURES = 5

# XGBoost hyperparams
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# DeepLabV3+ hyperparams
DL_ENCODER = "efficientnet-b0"
DL_ENCODER_WEIGHTS = "imagenet"
DL_CLASSES = 1
DL_FOCAL_ALPHA = 0.75
DL_FOCAL_GAMMA = 2.0
DL_LR = 1e-4

# Patch config
S2_PATCH_SIZE = 128   # 128px at 50m scale for S2
S2_SCALE = 50
L8_PATCH_SIZE = 64    # 64px at 30m scale for L8
L8_SCALE = 30

# Training config
N_PATCHES = 1000
BATCH_SIZE = 16
EPOCHS = 15
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ESA WorldCover mangrove class
MANGROVE_CLASS = 95

# Pixel-level ML subsampling
PIXEL_SUBSAMPLE = 100_000
