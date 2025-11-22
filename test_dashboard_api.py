"""
Quick test script to verify the realtime API endpoints
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_realtime_visualizations():
    """Test the realtime visualizations endpoint"""
    try:
        print("Testing /realtime/visualizations endpoint...")
        response = requests.get(f"{API_BASE_URL}/realtime/visualizations")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Status: {data.get('status')}")
            
            if 'plots' in data:
                plots = data['plots']
                print(f"\n📊 Available plots:")
                for plot_name in plots.keys():
                    plot_data_length = len(plots[plot_name]) if plots[plot_name] else 0
                    print(f"  - {plot_name}: {plot_data_length} bytes")
            
            if 'metrics_summary' in data:
                metrics = data['metrics_summary']
                print(f"\n📈 Metrics Summary:")
                print(f"  - Total Processed: {metrics.get('total_processed', 0)}")
                print(f"  - Attack Rate: {metrics.get('attack_rate', 0)}%")
                print(f"  - Last Update: {metrics.get('last_update', 'N/A')}")
            
            return True
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend server is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_realtime_metrics():
    """Test the realtime metrics endpoint"""
    try:
        print("\n\nTesting /realtime/metrics endpoint...")
        response = requests.get(f"{API_BASE_URL}/realtime/metrics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Status: {data.get('status')}")
            
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"\n📊 Detailed Metrics:")
                print(f"  - Total Processed: {metrics.get('total_processed', 0)}")
                print(f"  - Attack Count: {metrics.get('attack_count', 0)}")
                print(f"  - Normal Count: {metrics.get('normal_count', 0)}")
                print(f"  - Attack Rate: {metrics.get('attack_rate_percent', 0):.2f}%")
            
            return True
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing SurakshaNET Dashboard API Integration")
    print("="*60)
    
    viz_ok = test_realtime_visualizations()
    metrics_ok = test_realtime_metrics()
    
    print("\n" + "="*60)
    if viz_ok and metrics_ok:
        print("✅ All tests passed! Dashboard should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the backend server.")
    print("="*60)
