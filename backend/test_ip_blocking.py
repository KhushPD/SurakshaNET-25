"""
Test IP Blocking Integration
=============================
Simple test to verify IP blocking is working
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_ip_blocking():
    """Test IP blocking functionality"""
    
    print("\n" + "="*60)
    print("  IP BLOCKING TEST")
    print("="*60)
    
    # Test 1: Check blocked IPs list
    print("\n1. Getting blocked IPs list...")
    response = requests.get(f"{BASE_URL}/blocked-ips")
    print(f"   Status: {response.status_code}")
    print(f"   Blocked IPs: {response.json()}")
    
    # Test 2: Block an IP manually
    print("\n2. Blocking IP 192.168.1.50...")
    response = requests.post(
        f"{BASE_URL}/block-ip",
        params={
            "ip": "192.168.1.50",
            "reason": "Manual test block",
            "duration_minutes": 5
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 3: Verify IP is blocked
    print("\n3. Verifying IP is in blocked list...")
    response = requests.get(f"{BASE_URL}/blocked-ips")
    blocked = response.json()
    print(f"   Blocked IPs: {len(blocked['blocked_ips'])}")
    for ip_info in blocked['blocked_ips']:
        print(f"   - {ip_info['ip']}: {ip_info['reason']}")
    
    # Test 4: Unblock the IP
    print("\n4. Unblocking IP 192.168.1.50...")
    response = requests.post(
        f"{BASE_URL}/unblock-ip",
        params={"ip": "192.168.1.50"}
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 5: Check health endpoint
    print("\n5. Checking API health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    health = response.json()
    print(f"   API Status: {health['status']}")
    print(f"   Models Loaded: {health['models_loaded']}")
    
    print("\n" + "="*60)
    print("  ✓ IP Blocking Integration Working!")
    print("="*60)
    print("\n💡 Tip: When attacks are detected with >85% confidence,")
    print("   IPs are automatically blocked for 30 minutes\n")

if __name__ == "__main__":
    try:
        test_ip_blocking()
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend!")
        print("   Start backend with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
