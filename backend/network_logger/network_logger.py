"""
Network Logger Module
Captures and stores network traffic logs in real-time
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import threading
from collections import deque

class NetworkLogger:
    def __init__(self, log_file: str = "network_logs.json", max_logs: int = 10000):
        """
        Initialize the Network Logger
        
        Args:
            log_file: Path to store logs
            max_logs: Maximum number of logs to keep in memory
        """
        self.log_file = Path(log_file)
        self.max_logs = max_logs
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()
        self._load_existing_logs()
        
    def _load_existing_logs(self):
        """Load existing logs from file if available"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.logs = deque(data, maxlen=self.max_logs)
            except Exception as e:
                print(f"Error loading logs: {e}")
                self.logs = deque(maxlen=self.max_logs)
    
    def log_request(self, 
                    source_ip: str,
                    destination_ip: str,
                    source_port: int,
                    destination_port: int,
                    protocol: str,
                    request_type: str,
                    response_code: Optional[int] = None,
                    packet_size: int = 0,
                    duration: float = 0.0,
                    additional_data: Optional[Dict] = None) -> Dict:
        """
        Log a network request
        
        Returns:
            Dict containing the logged entry
        """
        timestamp = datetime.now()
        
        log_entry = {
            "id": self._generate_id(source_ip, destination_ip, timestamp),
            "timestamp": timestamp.isoformat(),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
            "request_type": request_type,
            "response_code": response_code,
            "packet_size": packet_size,
            "duration": duration,
            "status": "logged"
        }
        
        if additional_data:
            log_entry.update(additional_data)
        
        # Add hash for blockchain-style integrity
        log_entry["hash"] = self._calculate_hash(log_entry)
        
        with self.lock:
            self.logs.append(log_entry)
        
        return log_entry
    
    def log_intrusion(self,
                     source_ip: str,
                     destination_ip: str,
                     intrusion_type: str,
                     confidence: float,
                     action: str,
                     model_prediction: str,
                     features: Optional[Dict] = None) -> Dict:
        """
        Log a detected intrusion
        
        Returns:
            Dict containing the intrusion log entry
        """
        timestamp = datetime.now()
        
        intrusion_entry = {
            "id": self._generate_id(source_ip, destination_ip, timestamp),
            "timestamp": timestamp.isoformat(),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "intrusion_type": intrusion_type,
            "confidence": confidence,
            "action": action,
            "model_prediction": model_prediction,
            "severity": self._calculate_severity(confidence),
            "status": "detected"
        }
        
        if features:
            intrusion_entry["features"] = features
        
        # Add hash for blockchain-style integrity
        intrusion_entry["hash"] = self._calculate_hash(intrusion_entry)
        
        # Link to previous log (blockchain-style)
        if len(self.logs) > 0:
            intrusion_entry["previous_hash"] = self.logs[-1].get("hash", "")
        
        with self.lock:
            self.logs.append(intrusion_entry)
        
        return intrusion_entry
    
    def get_logs(self, 
                 limit: int = 100, 
                 filter_type: Optional[str] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None) -> List[Dict]:
        """
        Retrieve logs with optional filtering
        
        Args:
            limit: Maximum number of logs to return
            filter_type: Filter by status ('logged', 'detected', or None for all)
            start_time: ISO format start time
            end_time: ISO format end time
        """
        with self.lock:
            filtered_logs = list(self.logs)
        
        if filter_type:
            filtered_logs = [log for log in filtered_logs if log.get("status") == filter_type]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log["timestamp"] <= end_time]
        
        return filtered_logs[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get statistics about logged traffic"""
        with self.lock:
            logs_list = list(self.logs)
        
        total_logs = len(logs_list)
        intrusions = [log for log in logs_list if log.get("status") == "detected"]
        benign = [log for log in logs_list if log.get("status") == "logged"]
        
        return {
            "total_logs": total_logs,
            "total_intrusions": len(intrusions),
            "total_benign": len(benign),
            "intrusion_rate": len(intrusions) / total_logs if total_logs > 0 else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def save_to_file(self):
        """Persist logs to file"""
        with self.lock:
            logs_list = list(self.logs)
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(logs_list, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving logs: {e}")
            return False
    
    def clear_logs(self):
        """Clear all logs"""
        with self.lock:
            self.logs.clear()
        self.save_to_file()
    
    def _generate_id(self, source_ip: str, destination_ip: str, timestamp: datetime) -> str:
        """Generate unique ID for log entry"""
        data = f"{source_ip}{destination_ip}{timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _calculate_hash(self, log_entry: Dict) -> str:
        """Calculate SHA-256 hash for blockchain-style integrity"""
        # Remove hash field if it exists to avoid circular hashing
        entry_copy = {k: v for k, v in log_entry.items() if k != "hash"}
        entry_string = json.dumps(entry_copy, sort_keys=True)
        return hashlib.sha256(entry_string.encode()).hexdigest()
    
    def _calculate_severity(self, confidence: float) -> str:
        """Calculate severity level based on confidence"""
        if confidence >= 0.9:
            return "CRITICAL"
        elif confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def verify_chain_integrity(self) -> bool:
        """Verify blockchain-style chain integrity"""
        with self.lock:
            logs_list = list(self.logs)
        
        for i in range(1, len(logs_list)):
            if logs_list[i].get("previous_hash") != logs_list[i-1].get("hash"):
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    logger = NetworkLogger()
    
    # Log a normal request
    normal_log = logger.log_request(
        source_ip="192.168.1.100",
        destination_ip="93.184.216.34",
        source_port=54321,
        destination_port=80,
        protocol="TCP",
        request_type="GET",
        response_code=200,
        packet_size=1024,
        duration=0.15
    )
    print(f"Normal request logged: {normal_log['id']}")
    
    # Log an intrusion
    intrusion_log = logger.log_intrusion(
        source_ip="10.0.0.50",
        destination_ip="192.168.1.1",
        intrusion_type="Port Scan",
        confidence=0.95,
        action="Block Source IP",
        model_prediction="Intrusion",
        features={
            "packet_count": 1000,
            "duration": 5.0,
            "protocol": "TCP",
            "flag_count": 500
        }
    )
    print(f"Intrusion logged: {intrusion_log['id']}")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Save to file
    logger.save_to_file()
    print("\nLogs saved to file")
    
    # Verify chain integrity
    is_valid = logger.verify_chain_integrity()
    print(f"Chain integrity: {'Valid' if is_valid else 'Invalid'}")