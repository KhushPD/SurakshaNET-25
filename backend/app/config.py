"""
Configuration settings for the FastAPI backend.
Centralizes all paths and settings.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

# Model directories
MODELS_DIR = BACKEND_DIR / "models" / "trained"

# Upload and output directories
UPLOADS_DIR = BACKEND_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model file names
MODEL_FILES = {
    "rf_binary": "random_forest_binary.joblib",
    "rf_multiclass": "random_forest_multiclass.joblib",
    "xgb_binary": "xgboost_binary.joblib",
    "xgb_multiclass": "xgboost_multiclass.joblib"
}

# API settings
API_TITLE = "SurakshaNET - Network Intrusion Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Network Intrusion Detection System API

This API accepts network traffic data in CSV format and returns:
- Binary classification (Normal vs Attack)
- Multi-class classification (Normal, DoS, Probe, R2L, U2R)
- Detailed metrics and visualizations
- Comprehensive reports
"""

# File upload limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".csv"}

# Processing limits
MAX_ROWS_FOR_FULL_ANALYSIS = 10000  # Full processing for datasets up to 10K rows
MAX_ROWS_FOR_VISUALIZATION = 5000   # Generate all visualizations for up to 5K rows
PREDICTION_BATCH_SIZE = 1000        # Process predictions in batches

# Feature adaptation settings
ENABLE_FEATURE_ADAPTATION = True    # Allow flexible feature counts
MIN_FEATURES_REQUIRED = 3           # Minimum features needed for prediction
FEATURE_ENGINEERING_ENABLED = True  # Enable automatic feature engineering

# Label mappings
BINARY_LABELS = {0: "Normal", 1: "Attack"}
MULTICLASS_LABELS = {
    0: "Normal",
    1: "DoS",
    2: "Probe", 
    3: "R2L",
    4: "U2R"
}
    