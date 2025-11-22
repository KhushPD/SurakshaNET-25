"""
STRESS TEST 1 - EASY
Pattern: Normal Traffic → Light Attack Wave → Normal Recovery
Duration: ~60 seconds
"""
import requests
import time
import random
import csv
from pathlib import Path

BASE_URL = "http://localhost:8000"


def load_dataset(dataset_path: str):
    """Load dataset and separate by attack/normal"""
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    
    label_idx = header.index('label') if 'label' in header else -1
    normal_rows = []
    attack_rows = []
    
    for row in all_rows:
        label = row[label_idx].strip().upper()
        if label in ["NORMAL", "BENIGN"]:
            normal_rows.append(row)
        else:
            attack_rows.append(row)
    
    return header, normal_rows, attack_rows


def send_batch(header, sample_rows, batch_num, phase):
    """Send batch to backend"""
    temp_file = "temp_batch.csv"
    with open(temp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sample_rows)
    
    with open(temp_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/realtime/process-upload", files=files)
    
    if response.status_code == 200:
        data = response.json()
        metrics = data.get('metrics', {})
        print(f"[{phase}] Batch #{batch_num}: Total={metrics.get('total_processed', 0)} | "
              f"Attacks={metrics.get('attack_count', 0)} | "
              f"Rate={metrics.get('attack_rate_percent', 0):.1f}%")
        return True
    return False


def run_stress_test_1(dataset_path: str):
    """
    STRESS TEST 1 - EASY
    Pattern: Normal (20s) → Attack Wave (20s) → Normal Recovery (20s)
    Total Duration: ~60 seconds
    """
    print("\n" + "="*70)
    print("🟢 STRESS TEST 1 - EASY")
    print("Pattern: Normal → Attack Wave → Normal Recovery")
    print("Duration: ~60 seconds")
    print("="*70 + "\n")
    
    header, normal_rows, attack_rows = load_dataset(dataset_path)
    batch_num = 0
    
    # Phase 1: Normal Traffic (20s)
    print("📊 Phase 1: Normal Traffic (20s)")
    for _ in range(4):
        batch_num += 1
        sample = random.sample(normal_rows, min(30, len(normal_rows)))
        send_batch(header, sample, batch_num, "NORMAL")
        time.sleep(5)
    
    # Phase 2: Attack Wave (20s)
    print("\n🔴 Phase 2: Attack Wave (20s)")
    for _ in range(4):
        batch_num += 1
        sample = random.sample(attack_rows, min(40, len(attack_rows)))
        send_batch(header, sample, batch_num, "ATTACK")
        time.sleep(5)
    
    # Phase 3: Normal Recovery (20s)
    print("\n📊 Phase 3: Normal Recovery (20s)")
    for _ in range(4):
        batch_num += 1
        sample = random.sample(normal_rows, min(30, len(normal_rows)))
        send_batch(header, sample, batch_num, "RECOVERY")
        time.sleep(5)
    
    print("\n✅ Stress Test 1 Complete! (~60s)\n")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / "dataset" / "nsl_kdd_dataset.csv"
    if dataset_path.exists():
        run_stress_test_1(str(dataset_path))
    else:
        print(f"Dataset not found: {dataset_path}")
