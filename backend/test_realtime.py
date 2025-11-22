"""
Test Script for Real-Time Monitoring System
===========================================
Quick tests to verify the real-time monitoring backend is working correctly.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_health():
    """Test basic health endpoint."""
    print_section("1. Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_start_monitoring():
    """Test starting real-time monitoring."""
    print_section("2. Starting Real-Time Monitoring")
    response = requests.post(f"{BASE_URL}/realtime/start?use_simulation=true")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_get_status():
    """Test getting monitoring status."""
    print_section("3. Getting Monitoring Status")
    response = requests.get(f"{BASE_URL}/realtime/status")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Is Running: {data.get('is_running', False)}")
    print(f"Total Processed: {data.get('metrics', {}).get('total_processed', 0)}")
    print(f"Attack Rate: {data.get('metrics', {}).get('attack_rate_percent', 0)}%")
    return response.status_code == 200

def test_get_metrics():
    """Test getting detailed metrics."""
    print_section("4. Getting Detailed Metrics")
    response = requests.get(f"{BASE_URL}/realtime/metrics")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if "metrics" in data:
        metrics = data["metrics"]
        print(f"Total Processed: {metrics.get('total_processed', 0)}")
        print(f"Total Attacks: {metrics.get('total_attacks', 0)}")
        print(f"Total Normal: {metrics.get('total_normal', 0)}")
        print(f"Attack Rate: {metrics.get('attack_rate_percent', 0):.2f}%")
        print(f"Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
        print(f"Last Update: {metrics.get('last_update', 'N/A')}")
        
        print("\nAttack Type Distribution:")
        attack_counts = metrics.get('attack_type_counts_all', {})
        labels = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
        for attack_id, count in attack_counts.items():
            print(f"  {labels.get(int(attack_id), 'Unknown')}: {count}")
    
    return response.status_code == 200

def test_get_attack_types():
    """Test getting attack type breakdown."""
    print_section("5. Getting Attack Type Breakdown")
    response = requests.get(f"{BASE_URL}/realtime/attack-types")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if "attack_types" in data:
        print(f"Total Processed: {data.get('total_processed', 0)}")
        print("\nAttack Types:")
        for attack in data["attack_types"]:
            print(f"  {attack['type']:10s}: {attack['count']:5d} ({attack['percentage']:5.2f}%)")
    
    return response.status_code == 200

def test_get_visualizations():
    """Test getting visualizations."""
    print_section("6. Getting Visualizations")
    response = requests.get(f"{BASE_URL}/realtime/visualizations")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if "plots" in data:
        plots = data["plots"]
        print(f"Number of plots generated: {len(plots)}")
        print("Available plots:")
        for plot_name in plots.keys():
            plot_size = len(plots[plot_name])
            print(f"  - {plot_name}: {plot_size} bytes (base64)")
        
        if "metrics_summary" in data:
            summary = data["metrics_summary"]
            print(f"\nMetrics Summary:")
            print(f"  Total Processed: {summary.get('total_processed', 0)}")
            print(f"  Attack Rate: {summary.get('attack_rate', 0):.2f}%")
            print(f"  Last Update: {summary.get('last_update', 'N/A')}")
    
    return response.status_code == 200

def test_stop_monitoring():
    """Test stopping monitoring."""
    print_section("7. Stopping Real-Time Monitoring")
    response = requests.post(f"{BASE_URL}/realtime/stop")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_reset_metrics():
    """Test resetting metrics."""
    print_section("8. Resetting Metrics")
    response = requests.post(f"{BASE_URL}/realtime/reset")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  REAL-TIME MONITORING SYSTEM TEST SUITE")
    print("  Make sure the backend is running on http://localhost:8000")
    print("="*70)
    
    results = []
    
    try:
        # Test 1: Health check
        results.append(("Health Check", test_health()))
        
        # Test 2: Start monitoring
        results.append(("Start Monitoring", test_start_monitoring()))
        
        # Wait for some data to accumulate
        print("\n⏳ Waiting 5 seconds for data collection...")
        time.sleep(5)
        
        # Test 3: Get status
        results.append(("Get Status", test_get_status()))
        
        # Test 4: Get metrics
        results.append(("Get Metrics", test_get_metrics()))
        
        # Test 5: Get attack types
        results.append(("Get Attack Types", test_get_attack_types()))
        
        # Test 6: Get visualizations
        results.append(("Get Visualizations", test_get_visualizations()))
        
        # Test 7: Stop monitoring
        results.append(("Stop Monitoring", test_stop_monitoring()))
        
        # Test 8: Reset metrics
        results.append(("Reset Metrics", test_reset_metrics()))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to backend!")
        print("Make sure the backend is running:")
        print("  cd backend")
        print("  uvicorn app.main:app --reload")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Real-time monitoring system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
