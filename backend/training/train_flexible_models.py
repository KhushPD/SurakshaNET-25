"""
Flexible Model Training Script
===============================
Train models that work with any feature count through feature adaptation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_flexible_models(data_path: str, output_dir: str):
    """
    Train models with feature metadata for flexible inference.
    
    Args:
        data_path: Path to training dataset CSV
        output_dir: Directory to save trained models
    """
    logger.info("="*70)
    logger.info("FLEXIBLE MODEL TRAINING")
    logger.info("="*70)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Prepare features
    label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
    feature_columns = [col for col in df.columns if col not in label_columns]
    
    X = df[feature_columns].values
    feature_names = feature_columns
    
    logger.info(f"Features: {len(feature_names)}")
    
    # Prepare labels
    if 'label_binary' in df.columns:
        y_binary = df['label_binary'].values
    else:
        # Create binary labels (0=normal, 1=attack)
        y_binary = (df['label'] != 'normal').astype(int).values
    
    if 'label_encoded' in df.columns:
        y_multi = df['label_encoded'].values
    else:
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_multi = le.fit_transform(df['label'].values)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X_scaled, y_binary, y_multi, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train models
    models = {}
    
    # 1. XGBoost Binary
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost Binary Classifier...")
    xgb_binary = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_binary.fit(X_train, y_train_bin)
    y_pred = xgb_binary.predict(X_test)
    acc = accuracy_score(y_test_bin, y_pred)
    logger.info(f"✓ Binary Accuracy: {acc:.4f}")
    models['xgb_binary'] = xgb_binary
    
    # 2. XGBoost Multiclass
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost Multiclass Classifier...")
    xgb_multi = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_multi.fit(X_train, y_train_multi)
    y_pred = xgb_multi.predict(X_test)
    acc = accuracy_score(y_test_multi, y_pred)
    logger.info(f"✓ Multiclass Accuracy: {acc:.4f}")
    models['xgb_multiclass'] = xgb_multi
    
    # 3. Random Forest Binary
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest Binary Classifier...")
    rf_binary = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_binary.fit(X_train, y_train_bin)
    y_pred = rf_binary.predict(X_test)
    acc = accuracy_score(y_test_bin, y_pred)
    logger.info(f"✓ Binary Accuracy: {acc:.4f}")
    models['rf_binary'] = rf_binary
    
    # 4. Random Forest Multiclass
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest Multiclass Classifier...")
    rf_multi = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_multi.fit(X_train, y_train_multi)
    y_pred = rf_multi.predict(X_test)
    acc = accuracy_score(y_test_multi, y_pred)
    logger.info(f"✓ Multiclass Accuracy: {acc:.4f}")
    models['rf_multiclass'] = rf_multi
    
    # Save models
    logger.info("\n" + "="*50)
    logger.info("Saving models...")
    
    model_files = {
        'xgb_binary': 'xgboost_binary.joblib',
        'xgb_multiclass': 'xgboost_multiclass.joblib',
        'rf_binary': 'random_forest_binary.joblib',
        'rf_multiclass': 'random_forest_multiclass.joblib'
    }
    
    for key, filename in model_files.items():
        path = output_path / filename
        joblib.dump(models[key], path)
        logger.info(f"✓ Saved: {filename}")
    
    # Save feature metadata
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'scaler': scaler
    }
    metadata_path = output_path / 'feature_metadata.joblib'
    joblib.dump(metadata, metadata_path)
    logger.info(f"✓ Saved: feature_metadata.joblib")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Models saved to: {output_path}")
    logger.info(f"Feature count: {len(feature_names)}")
    logger.info("Models are ready for flexible inference!")


if __name__ == "__main__":
    # Default paths
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "dataset" / "nsl_kdd_dataset.csv"
    output_dir = base_dir / "backend" / "models" / "trained"
    
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Please ensure nsl_kdd_dataset.csv exists in the dataset directory")
    else:
        train_flexible_models(str(data_path), str(output_dir))
