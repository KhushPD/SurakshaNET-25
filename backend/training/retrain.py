"""
Quick Retrain Script
====================
Retrain models with feature metadata for flexible inference.

Usage:
    python retrain.py
    
or from backend directory:
    cd backend
    python -m training.retrain
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_flexible_models import train_flexible_models

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "dataset" / "nsl_kdd_dataset.csv"
    output_dir = base_dir / "backend" / "models" / "trained"
    
    print("\n" + "="*70)
    print("QUICK MODEL RETRAIN")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    if not data_path.exists():
        print(f"❌ ERROR: Dataset not found at {data_path}")
        print("\nAvailable datasets:")
        dataset_dir = base_dir / "dataset"
        if dataset_dir.exists():
            for csv_file in dataset_dir.glob("*.csv"):
                print(f"  - {csv_file.name}")
        sys.exit(1)
    
    # Start training
    train_flexible_models(str(data_path), str(output_dir))
    
    print("\n✅ Models retrained successfully!")
    print("🔄 Restart the API server to load new models.")
