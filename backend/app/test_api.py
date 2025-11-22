"""
API Test Script
===============
Tests the FastAPI backend with a sample CSV file.
"""

import requests
import pandas as pd
import json
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:8000"
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "cleaned" / "nsl_kdd_cleaned.csv"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        print("✓ Health check passed")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"✗ Health check failed: {response.status_code}")
    
    print()


def test_prediction():
    """Test the prediction endpoint with sample data."""
    print("Testing prediction endpoint...")
    
    # Load sample data (first 100 rows)
    print(f"Loading test data from: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    sample_df = df.head(100)
    
    # Save to temporary CSV
    temp_csv = Path("temp_test.csv")
    sample_df.to_csv(temp_csv, index=False)
    print(f"Created test file with {len(sample_df)} samples")
    
    # Send to API
    print("Sending request to API...")
    with open(temp_csv, 'rb') as f:
        files = {'file': ('test.csv', f, 'text/csv')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    # Clean up
    temp_csv.unlink()
    
    if response.status_code == 200:
        print("✓ Prediction successful!")
        result = response.json()
        
        # Print summary
        print("\n📊 PREDICTION SUMMARY:")
        summary = result['summary']
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Normal: {summary['normal_percentage']}%")
        print(f"  Attack: {summary['attack_percentage']}%")
        
        print("\n🔵 Binary Distribution:")
        for dist in summary['binary_distribution']:
            print(f"  {dist['label']}: {dist['count']} ({dist['percentage']}%)")
        
        print("\n🟣 Multi-Class Distribution:")
        for dist in summary['multiclass_distribution']:
            print(f"  {dist['label']}: {dist['count']} ({dist['percentage']}%)")
        
        print(f"\n📈 Generated {len(result['plots'])} visualization plots")
        print(f"📝 Returned {len(result['predictions'])} detailed predictions")
        
        # Show first 5 predictions
        print("\n🔍 Sample Predictions (first 5):")
        for pred in result['predictions'][:5]:
            print(f"  Sample {pred['sample_id']}: "
                  f"Binary={pred['binary_prediction']} ({pred['binary_confidence']:.2%}), "
                  f"Multi={pred['multiclass_prediction']} ({pred['multiclass_confidence']:.2%})")
    else:
        print(f"✗ Prediction failed: {response.status_code}")
        print(response.text)
    
    print()


def test_models_endpoint():
    """Test the models info endpoint."""
    print("Testing models endpoint...")
    response = requests.get(f"{API_URL}/models")
    
    if response.status_code == 200:
        print("✓ Models endpoint working")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"✗ Models endpoint failed: {response.status_code}")
    
    print()


def main():
    """Run all tests."""
    print("="*70)
    print("SurakshaNET API TEST SUITE")
    print("="*70)
    print()
    
    try:
        test_health_check()
        test_models_endpoint()
        test_prediction()
        
        print("="*70)
        print("ALL TESTS COMPLETED!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure the server is running:")
        print(f"  uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")


if __name__ == "__main__":
    main()
