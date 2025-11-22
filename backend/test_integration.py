"""
Integration Test Script
=======================
Tests the complete workflow: CSV upload -> ML prediction -> Response with plots
"""

import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_FILE = Path("uploads/test_sample.csv")

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("Testing /health endpoint...")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {data['status']}")
        print(f"✓ Message: {data['message']}")
        print(f"✓ Models Loaded:")
        for model, loaded in data['models_loaded'].items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model}")
    else:
        print(f"✗ Health check failed: {response.text}")
    
    print()
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint with sample CSV"""
    print("=" * 70)
    print("Testing /predict endpoint with test_sample.csv...")
    print("=" * 70)
    
    if not TEST_FILE.exists():
        print(f"✗ Test file not found: {TEST_FILE}")
        return False
    
    # Read and upload file
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('test_sample.csv', f, 'text/csv')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Print summary
        summary = data['summary']
        print(f"\n✓ Analysis Complete!")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Attack %: {summary['attack_percentage']:.2f}%")
        print(f"  Normal %: {summary['normal_percentage']:.2f}%")
        
        print(f"\n✓ Binary Distribution:")
        for dist in summary['binary_distribution']:
            print(f"  {dist['label']}: {dist['count']} ({dist['percentage']:.1f}%)")
        
        print(f"\n✓ Multiclass Distribution:")
        for dist in summary['multiclass_distribution']:
            if dist['count'] > 0:
                print(f"  {dist['label']}: {dist['count']} ({dist['percentage']:.1f}%)")
        
        print(f"\n✓ Predictions: {len(data['predictions'])} samples")
        print(f"✓ Plots Generated: {len(data['plots'])} visualizations")
        for plot_name in data['plots'].keys():
            print(f"  - {plot_name}")
        
        print(f"\n✓ Message: {data['message']}")
        
        # Sample predictions
        print(f"\n✓ Sample Predictions (first 5):")
        for pred in data['predictions'][:5]:
            print(f"  Sample #{pred['sample_id']}: "
                  f"{pred['binary_prediction']} ({pred['binary_confidence']:.2%}) | "
                  f"{pred['multiclass_prediction']} ({pred['multiclass_confidence']:.2%})")
        
        return True
    else:
        error = response.json() if response.headers.get('content-type') == 'application/json' else response.text
        print(f"✗ Prediction failed: {error}")
        return False


def test_models():
    """Test models info endpoint"""
    print("\n" + "=" * 70)
    print("Testing /models endpoint...")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/models")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Primary Model: {data['primary_model']}")
        print(f"✓ Binary Classes: {', '.join(data['classes']['binary'])}")
        print(f"✓ Multiclass Classes: {', '.join(data['classes']['multiclass'])}")
    else:
        print(f"✗ Failed: {response.text}")
    
    return response.status_code == 200


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SURAKSHNET INTEGRATION TEST")
    print("=" * 70)
    print()
    
    results = {
        'Health Check': test_health(),
        'Models Info': test_models(),
        'Prediction': test_predict()
    }
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Integration successful!")
    else:
        print("✗ SOME TESTS FAILED - Check output above")
    print("=" * 70)
