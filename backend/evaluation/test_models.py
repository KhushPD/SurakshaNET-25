"""
Model Testing and Comparison Script
====================================
This script tests the trained models and compares their performance.

Usage:
    python test_models.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def test_models():
    """
    Test all trained models with sample data.
    
    What this does:
    1. Loads the test dataset
    2. Loads all 4 trained models
    3. Makes predictions
    4. Compares results
    """
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "dataset" / "cleaned" / "nsl_kdd_cleaned.csv"
    model_dir = project_root / "backend" / "models" / "trained"
    
    logger.info("="*70)
    logger.info("MODEL TESTING AND COMPARISON")
    logger.info("="*70)
    
    # Load test data (take last 100 samples for quick test)
    logger.info("\nLoading test data...")
    df = pd.read_csv(data_path)
    
    # Get features and labels
    label_columns = ['label', 'label_original', 'label_binary', 'label_encoded']
    feature_columns = [col for col in df.columns if col not in label_columns]
    
    # Take last 100 samples for testing
    test_data = df.tail(100)
    X_test = test_data[feature_columns].values
    y_test_binary = test_data['label_binary'].values
    y_test_multi = test_data['label_encoded'].values
    y_test_labels = test_data['label'].values
    
    logger.info(f"Test samples: {len(test_data)}")
    logger.info(f"Features: {len(feature_columns)}")
    
    # Load models
    logger.info("\nLoading trained models...")
    models = {
        'RF Binary': load_model(model_dir / "random_forest_binary.joblib"),
        'RF Multi-Class': load_model(model_dir / "random_forest_multiclass.joblib"),
        'XGB Binary': load_model(model_dir / "xgboost_binary.joblib"),
        'XGB Multi-Class': load_model(model_dir / "xgboost_multiclass.joblib")
    }
    logger.info(f"Loaded {len(models)} models successfully")
    
    # Test binary classification models
    logger.info("\n" + "="*70)
    logger.info("BINARY CLASSIFICATION (Normal vs Attack)")
    logger.info("="*70)
    
    for model_name in ['RF Binary', 'XGB Binary']:
        model = models[model_name]
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test_binary).mean()
        
        # Count predictions
        normal_count = (predictions == 0).sum()
        attack_count = (predictions == 1).sum()
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Predicted Normal: {normal_count}")
        logger.info(f"  Predicted Attack: {attack_count}")
    
    # Test multi-class models
    logger.info("\n" + "="*70)
    logger.info("MULTI-CLASS CLASSIFICATION")
    logger.info("="*70)
    
    class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    
    for model_name in ['RF Multi-Class', 'XGB Multi-Class']:
        model = models[model_name]
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test_multi).mean()
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Count predictions per class
        for i, class_name in enumerate(class_names):
            count = (predictions == i).sum()
            logger.info(f"  Predicted {class_name}: {count}")
    
    # Show sample predictions
    logger.info("\n" + "="*70)
    logger.info("SAMPLE PREDICTIONS (First 10 samples)")
    logger.info("="*70)
    
    rf_binary_pred = models['RF Binary'].predict(X_test[:10])
    xgb_multi_pred = models['XGB Multi-Class'].predict(X_test[:10])
    
    logger.info("\n{:<15} {:<15} {:<20} {:<20}".format(
        "Actual Label", "RF Binary", "XGB Multi-Class", "Match?"
    ))
    logger.info("-"*70)
    
    for i in range(10):
        actual = y_test_labels[i]
        rf_pred = "Normal" if rf_binary_pred[i] == 0 else "Attack"
        xgb_pred = class_names[xgb_multi_pred[i]]
        match = "✓" if actual == xgb_pred else "✗"
        
        logger.info("{:<15} {:<15} {:<20} {:<20}".format(
            actual, rf_pred, xgb_pred, match
        ))
    
    logger.info("\n" + "="*70)
    logger.info("TESTING COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    test_models()
