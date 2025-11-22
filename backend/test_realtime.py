"""
Real-Time Traffic Generator with Stress Tests for Dashboard Testing
====================================================================
Runs beautiful network stress tests that generate traffic patterns
for the real-time dashboard to display.
"""
import sys
from pathlib import Path

# Add network_tester to path
sys.path.insert(0, str(Path(__file__).parent / "network_tester"))

from stress_test1 import run_stress_test_1
from stress_test2 import run_stress_test_2
from stress_test3 import run_stress_test_3
import requests
import time

BASE_URL = "http://localhost:8000"


def main():
    """Run stress tests."""
    print("\n" + "="*70)
    print("  REAL-TIME DASHBOARD - STRESS TEST SUITE")
    print("  Make sure backend is running on http://localhost:8000")
    print("  Make sure dashboard is open at http://localhost:5173")
    print("="*70)
    
    # Check if backend is accessible
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("\n❌ Backend not responding!")
            return
    except:
        print("\n❌ Cannot connect to backend!")
        print("Start backend with: uvicorn app.main:app --reload")
        return
    
    print("\n✅ Backend is running\n")
    
    # Find dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "nsl_kdd_dataset.csv"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"✅ Dataset found: {dataset_path.name}\n")
    
    # Ask user for test selection
    print("Select Stress Test:")
    print("1. Test 1 - Easy    (~60s)  Normal → Attack → Normal")
    print("2. Test 2 - Medium  (~90s)  Normal → Mixed → Spike → Normal")
    print("3. Test 3 - Hard    (~120s) Normal → Escalating → Peak → Chaos → Recovery")
    print("4. Run All Tests    (~270s) All three tests in sequence")
    
    choice = input("\nChoice (1-4) [default: 1]: ").strip() or "1"
    
    print("\n" + "="*70)
    input("Press ENTER to start stress test...")
    print()
    
    try:
        dataset = str(dataset_path)
        
        if choice == "1":
            run_stress_test_1(dataset)
        elif choice == "2":
            run_stress_test_2(dataset)
        elif choice == "3":
            run_stress_test_3(dataset)
        elif choice == "4":
            print("🚀 Running all 3 stress tests in sequence...\n")
            run_stress_test_1(dataset)
            time.sleep(5)
            run_stress_test_2(dataset)
            time.sleep(5)
            run_stress_test_3(dataset)
            print("\n🎉 All stress tests completed!\n")
        else:
            print("Invalid choice!")
            return
        
        print("💡 Check your dashboard at http://localhost:5173 to see the results!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Stress test stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
