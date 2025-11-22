#!/usr/bin/env python3
"""
Quick test script to verify backend API returns plots correctly
"""
import requests
import json

# Test file path
test_file = r"backend\uploads\test_sample.csv"

print("Testing backend API...")
print(f"Uploading file: {test_file}")

try:
    # Send file to backend
    with open(test_file, 'rb') as f:
        files = {'file': ('test_sample.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/predict', files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n=== SUMMARY ===")
        for key, value in data['summary'].items():
            print(f"  {key}: {value}")
        
        print("\n=== PLOTS ===")
        if 'plots' in data:
            for plot_name in data['plots'].keys():
                plot_data = data['plots'][plot_name]
                # Check if it's a base64 string
                if plot_data and plot_data.startswith('data:image'):
                    print(f"  ✓ {plot_name}: {len(plot_data)} bytes (base64 image)")
                else:
                    print(f"  ✗ {plot_name}: Invalid format")
        else:
            print("  No plots in response!")
        
        print(f"\n=== PREDICTIONS (first 5) ===")
        if data.get('predictions'):
            for i, pred in enumerate(data['predictions'][:5]):
                print(f"  {i+1}. {pred}")
        else:
            print("  No predictions in response")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")
