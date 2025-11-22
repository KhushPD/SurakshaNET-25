"""
STRESS TEST 3 - HARD
Pattern: Normal → Escalating Attacks → Peak → Chaos → Gradual Recovery
Duration: ~120 seconds
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


def run_stress_test_3(dataset_path: str):
    """
    STRESS TEST 3 - HARD
    Pattern: Normal → Escalating → Peak → Chaos → Gradual Recovery
    Total Duration: ~120 seconds
    """
    print("\n" + "="*70)
    print("🔴 STRESS TEST 3 - HARD")
    print("Pattern: Normal → Escalating → Peak → Chaos → Recovery")
    print("Duration: ~120 seconds")
    print("="*70 + "\n")
    
    header, normal_rows, attack_rows = load_dataset(dataset_path)
    batch_num = 0
    
    # Phase 1: Normal Baseline (15s)
    print("📊 Phase 1: Normal Baseline (15s)")
    for _ in range(3):
        batch_num += 1
        sample = random.sample(normal_rows, min(50, len(normal_rows)))
        send_batch(header, sample, batch_num, "NORMAL")
        time.sleep(5)
    
    # Phase 2: Escalating Attacks (30s)
    print("\n⚠️  Phase 2: Escalating Attack Pattern (30s)")
    attack_percentages = [20, 30, 40, 60, 75, 85]
    for pct in attack_percentages:
        batch_num += 1
        attack_count = pct
        normal_count = 100 - pct
        attack_sample = random.sample(attack_rows, min(attack_count, len(attack_rows)))
        normal_sample = random.sample(normal_rows, min(normal_count, len(normal_rows)))
        mixed = attack_sample + normal_sample
        random.shuffle(mixed)
        send_batch(header, mixed, batch_num, "ESCALATING")
        time.sleep(5)
    
    # Phase 3: Peak Attack Load (20s)
    print("\n🔥 Phase 3: Peak Attack Load (20s)")
    for _ in range(4):
        batch_num += 1
        sample = random.sample(attack_rows, min(80, len(attack_rows)))
        send_batch(header, sample, batch_num, "PEAK")
        time.sleep(5)
    
    # Phase 4: Chaotic Mixed (30s)
    print("\n⚡ Phase 4: Chaotic Mixed Traffic (30s)")
    for _ in range(6):
        batch_num += 1
        attack_count = random.randint(30, 70)
        normal_count = 100 - attack_count
        attack_sample = random.sample(attack_rows, min(attack_count, len(attack_rows)))
        normal_sample = random.sample(normal_rows, min(normal_count, len(normal_rows)))
        mixed = attack_sample + normal_sample
        random.shuffle(mixed)
        send_batch(header, mixed, batch_num, "CHAOS")
        time.sleep(5)
    
    # Phase 5: Gradual Recovery (25s)
    print("\n💚 Phase 5: Gradual Recovery (25s)")
    attack_percentages = [60, 40, 25, 15, 5]
    for pct in attack_percentages:
        batch_num += 1
        attack_count = pct
        normal_count = 100 - pct
        attack_sample = random.sample(attack_rows, min(attack_count, len(attack_rows)))
        normal_sample = random.sample(normal_rows, min(normal_count, len(normal_rows)))
        mixed = attack_sample + normal_sample
        random.shuffle(mixed)
        send_batch(header, mixed, batch_num, "RECOVERY")
        time.sleep(5)
    
    print("\n✅ Stress Test 3 Complete! (~120s)\n")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / "dataset" / "nsl_kdd_dataset.csv"
    if dataset_path.exists():
        run_stress_test_3(str(dataset_path))
    else:
        print(f"Dataset not found: {dataset_path}")
