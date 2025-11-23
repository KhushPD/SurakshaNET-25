"""
Real-Time Traffic Generator for Dashboard Testing
=================================================
Integrates with stress test scripts for comprehensive testing.
"""
import sys
from pathlib import Path

# Add stress test module to path
sys.path.insert(0, str(Path(__file__).parent / "network_tester"))

import requests
import time
from stress_test1 import run_stress_test_1
from stress_test2 import run_stress_test_2
from stress_test3 import run_stress_test_3

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
    print("2. Test 2 - Medium  (~90s)  Normal → Mixed → Spike → Recovery")
    print("3. Test 3 - Hard    (~120s) Escalating → Peak → Chaos → Recovery")
    print("4. All Tests        (~270s) Run all tests sequentially")
    print("="*70)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    # Start real-time monitoring
    print("\n🔄 Starting real-time monitoring...")
    try:
        response = requests.post(f"{BASE_URL}/realtime/start", 
                                params={"use_simulation": False})
        if response.status_code == 200:
            print("✅ Real-time monitoring started\n")
        else:
            print(f"⚠️  Warning: Could not start monitoring (status {response.status_code})")
    except Exception as e:
        print(f"❌ Error starting monitoring: {str(e)}")
        return
    
    # Wait for system to initialize
    time.sleep(2)
    
    # Run selected test
    try:
        if choice == "1":
            run_stress_test_1(str(dataset_path))
        elif choice == "2":
            run_stress_test_2(str(dataset_path))
        elif choice == "3":
            run_stress_test_3(str(dataset_path))
        elif choice == "4":
            print("\n🚀 Running ALL tests sequentially...\n")
            run_stress_test_1(str(dataset_path))
            time.sleep(3)
            run_stress_test_2(str(dataset_path))
            time.sleep(3)
            run_stress_test_3(str(dataset_path))
        else:
            print("❌ Invalid choice!")
            return
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {str(e)}")
    
    # Final metrics
    print("\n" + "="*70)
    print("📊 FINAL METRICS")
    print("="*70)
    try:
        response = requests.get(f"{BASE_URL}/realtime/metrics")
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('metrics', {})
            
            print(f"\n📈 Total Processed:    {metrics.get('total_processed', 0):,}")
            print(f"🔴 Attacks Detected:   {metrics.get('attack_count', 0):,}")
            print(f"🟢 Normal Traffic:     {metrics.get('normal_count', 0):,}")
            print(f"📊 Attack Rate:        {metrics.get('attack_rate_percent', 0):.2f}%")
            print(f"⚡ Packets/Second:     {metrics.get('packets_per_second', 0):.2f}")
            
            # Attack breakdown
            attack_counts = metrics.get('attack_type_counts_all', {})
            if attack_counts:
                print(f"\n🎯 Attack Types Detected:")
                for attack_type, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        print(f"   • {attack_type}: {count}")
        else:
            print(f"⚠️  Could not fetch metrics (status {response.status_code})")
    except Exception as e:
        print(f"❌ Error fetching metrics: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ Test Complete!")
    print("💡 Check dashboard at http://localhost:5173 for visualizations")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
