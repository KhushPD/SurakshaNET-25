"""
STRESS TEST 2 - MEDIUM
Pattern: Normal → Mixed Traffic → Attack Spike → Normal Recovery
Duration: ~90 seconds
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


def run_stress_test_2(dataset_path: str):
    """
    STRESS TEST 2 - MEDIUM
    Pattern: Normal → Mixed Traffic → Normal → Attack Spike → Recovery
    Total Duration: ~3 seconds (ultra-fast)
    """
    print("\n" + "="*70)
    print("🟡 STRESS TEST 2 - MEDIUM (ULTRA-FAST)")
    print("Pattern: Normal → Mixed → Spike")
    print("Duration: ~3 seconds")
    print("="*70 + "\n")
    
    header, normal_rows, attack_rows = load_dataset(dataset_path)
    batch_num = 0
    
    # Phase 1: Normal (0.5s)
    print("📊 Phase 1: Normal Traffic (0.5s)")
    batch_num += 1
    sample = random.sample(normal_rows, min(20, len(normal_rows)))
    send_batch(header, sample, batch_num, "NORMAL")
    time.sleep(0.1)
    
    # Phase 2: Mixed Traffic (1s)
    print("\n🟠 Phase 2: Mixed Traffic (1s)")
    batch_num += 1
    attack_sample = random.sample(attack_rows, min(15, len(attack_rows)))
    normal_sample = random.sample(normal_rows, min(10, len(normal_rows)))
    mixed = attack_sample + normal_sample
    random.shuffle(mixed)
    send_batch(header, mixed, batch_num, "MIXED")
    time.sleep(0.1)
    
    # Phase 3: Attack Spike (1s)
    print("\n🔴 Phase 3: Attack Spike (1s)")
    batch_num += 1
    sample = random.sample(attack_rows, min(30, len(attack_rows)))
    send_batch(header, sample, batch_num, "SPIKE")
    time.sleep(0.1)
    
    print("\n✅ Stress Test 2 Complete! (~3s)\n")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / "dataset" / "nsl_kdd_dataset.csv"
    if dataset_path.exists():
        run_stress_test_2(str(dataset_path))
    else:
        print(f"Dataset not found: {dataset_path}")
