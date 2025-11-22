"""
Real-Time Network Monitoring Service
=====================================
Continuously monitors network traffic, processes through ML models,
and updates dashboard metrics in real-time.

Flow:
1. Capture network traffic (simulated or real)
2. Process through ML models for prediction
3. Update real-time metrics and statistics
4. Store in circular buffer for dashboard visualization
5. Provide WebSocket/SSE endpoints for live updates
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque
import logging
import threading
from pathlib import Path

from app.ml_service import ml_service
from network_logger.network_logger import NetworkLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeMetrics:
    """
    Stores real-time metrics for dashboard visualization.
    Uses circular buffers to maintain recent history.
    """
    
    def __init__(self, max_history: int = 1000, window_size: int = 100):
        """
        Initialize real-time metrics storage.
        
        Args:
            max_history: Maximum number of predictions to keep in memory
            window_size: Size of rolling window for statistics
        """
        self.max_history = max_history
        self.window_size = window_size
        
        # Circular buffers for predictions
        self.binary_predictions: Deque[int] = deque(maxlen=max_history)
        self.multiclass_predictions: Deque[int] = deque(maxlen=max_history)
        self.confidence_scores: Deque[float] = deque(maxlen=max_history)
        self.timestamps: Deque[datetime] = deque(maxlen=max_history)
        
        # Current statistics
        self.total_processed = 0
        self.total_attacks = 0
        self.total_normal = 0
        
        # Attack type counters
        self.attack_type_counts = {
            0: 0,  # Normal
            1: 0,  # DoS
            2: 0,  # Probe
            3: 0,  # R2L
            4: 0   # U2R
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Last update time
        self.last_update = datetime.now()
        
    def add_prediction(self, binary_pred: int, multiclass_pred: int, 
                      confidence: float, timestamp: Optional[datetime] = None):
        """
        Add a new prediction to the metrics.
        
        Args:
            binary_pred: Binary prediction (0=Normal, 1=Attack)
            multiclass_pred: Multi-class prediction (0-4)
            confidence: Prediction confidence score
            timestamp: Optional timestamp (defaults to now)
        """
        with self.lock:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Add to buffers
            self.binary_predictions.append(binary_pred)
            self.multiclass_predictions.append(multiclass_pred)
            self.confidence_scores.append(confidence)
            self.timestamps.append(timestamp)
            
            # Update counters
            self.total_processed += 1
            if binary_pred == 1:
                self.total_attacks += 1
            else:
                self.total_normal += 1
            
            # Update attack type counter
            self.attack_type_counts[multiclass_pred] = \
                self.attack_type_counts.get(multiclass_pred, 0) + 1
            
            self.last_update = timestamp
    
    def add_batch_predictions(self, binary_preds: np.ndarray, 
                            multiclass_preds: np.ndarray,
                            confidences: np.ndarray):
        """
        Add multiple predictions at once (batch processing).
        
        Args:
            binary_preds: Array of binary predictions
            multiclass_preds: Array of multi-class predictions
            confidences: Array of confidence scores
        """
        timestamp = datetime.now()
        
        with self.lock:
            for bp, mp, conf in zip(binary_preds, multiclass_preds, confidences):
                self.binary_predictions.append(int(bp))
                self.multiclass_predictions.append(int(mp))
                self.confidence_scores.append(float(conf))
                self.timestamps.append(timestamp)
                
                self.total_processed += 1
                if bp == 1:
                    self.total_attacks += 1
                else:
                    self.total_normal += 1
                
                self.attack_type_counts[int(mp)] = \
                    self.attack_type_counts.get(int(mp), 0) + 1
            
            self.last_update = timestamp
    
    def get_current_stats(self) -> Dict:
        """
        Get current statistics for dashboard.
        
        Returns:
            Dictionary with current metrics
        """
        with self.lock:
            if len(self.binary_predictions) == 0:
                return self._get_empty_stats()
            
            # Calculate recent window statistics
            recent_window = min(self.window_size, len(self.binary_predictions))
            recent_binary = list(self.binary_predictions)[-recent_window:]
            recent_multiclass = list(self.multiclass_predictions)[-recent_window:]
            recent_confidence = list(self.confidence_scores)[-recent_window:]
            
            # Binary classification stats
            attack_count = sum(recent_binary)
            normal_count = recent_window - attack_count
            attack_rate = (attack_count / recent_window * 100) if recent_window > 0 else 0
            
            # Attack type distribution (recent window)
            recent_attack_types = {
                0: recent_multiclass.count(0),
                1: recent_multiclass.count(1),
                2: recent_multiclass.count(2),
                3: recent_multiclass.count(3),
                4: recent_multiclass.count(4)
            }
            
            # Confidence statistics
            avg_confidence = np.mean(recent_confidence) if recent_confidence else 0.0
            min_confidence = np.min(recent_confidence) if recent_confidence else 0.0
            max_confidence = np.max(recent_confidence) if recent_confidence else 0.0
            
            return {
                "total_processed": self.total_processed,
                "total_attacks": self.total_attacks,
                "total_normal": self.total_normal,
                "recent_window_size": recent_window,
                "recent_attack_count": attack_count,
                "recent_normal_count": normal_count,
                "attack_rate_percent": round(attack_rate, 2),
                "attack_type_counts_all": self.attack_type_counts.copy(),
                "attack_type_counts_recent": recent_attack_types,
                "avg_confidence": round(avg_confidence, 4),
                "min_confidence": round(min_confidence, 4),
                "max_confidence": round(max_confidence, 4),
                "last_update": self.last_update.isoformat(),
                "buffer_utilization": round(len(self.binary_predictions) / self.max_history * 100, 1)
            }
    
    def get_timeline_data(self, num_points: int = 50) -> Dict:
        """
        Get timeline data for attack rate visualization.
        
        Args:
            num_points: Number of time points to return
            
        Returns:
            Dictionary with timeline data
        """
        with self.lock:
            if len(self.binary_predictions) == 0:
                return {"timestamps": [], "attack_rates": [], "total_counts": []}
            
            # Split into time windows
            total_samples = len(self.binary_predictions)
            window_size = max(1, total_samples // num_points)
            
            timestamps = []
            attack_rates = []
            total_counts = []
            
            binary_list = list(self.binary_predictions)
            time_list = list(self.timestamps)
            
            for i in range(0, total_samples, window_size):
                window = binary_list[i:i+window_size]
                if window:
                    attack_count = sum(window)
                    attack_rate = (attack_count / len(window) * 100)
                    
                    timestamps.append(time_list[i].isoformat())
                    attack_rates.append(round(attack_rate, 2))
                    total_counts.append(len(window))
            
            return {
                "timestamps": timestamps,
                "attack_rates": attack_rates,
                "total_counts": total_counts
            }
    
    def get_confidence_distribution(self, bins: int = 20) -> Dict:
        """
        Get confidence score distribution for histogram.
        
        Args:
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with histogram data
        """
        with self.lock:
            if len(self.confidence_scores) == 0:
                return {"bin_edges": [], "counts": [], "mean": 0.0}
            
            conf_array = np.array(list(self.confidence_scores))
            counts, bin_edges = np.histogram(conf_array, bins=bins, range=(0, 1))
            
            return {
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist(),
                "mean": round(float(np.mean(conf_array)), 4)
            }
    
    def _get_empty_stats(self) -> Dict:
        """Return empty statistics structure."""
        return {
            "total_processed": 0,
            "total_attacks": 0,
            "total_normal": 0,
            "recent_window_size": 0,
            "recent_attack_count": 0,
            "recent_normal_count": 0,
            "attack_rate_percent": 0.0,
            "attack_type_counts_all": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "attack_type_counts_recent": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "last_update": datetime.now().isoformat(),
            "buffer_utilization": 0.0
        }
    
    def reset(self):
        """Reset all metrics and buffers."""
        with self.lock:
            self.binary_predictions.clear()
            self.multiclass_predictions.clear()
            self.confidence_scores.clear()
            self.timestamps.clear()
            
            self.total_processed = 0
            self.total_attacks = 0
            self.total_normal = 0
            
            self.attack_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            self.last_update = datetime.now()


class RealTimeMonitoringService:
    """
    Main real-time monitoring service.
    Captures traffic, processes through ML models, and updates metrics.
    """
    
    def __init__(self, network_logger: NetworkLogger):
        """
        Initialize real-time monitoring service.
        
        Args:
            network_logger: NetworkLogger instance for logging detections
        """
        self.metrics = RealTimeMetrics(max_history=1000, window_size=100)
        self.network_logger = network_logger
        self.ml_service = ml_service
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task = None
        
        # Traffic simulation parameters
        self.traffic_interval = 1.0  # seconds between traffic batches
        self.traffic_batch_size = 10  # number of samples per batch
        
        logger.info("Real-time monitoring service initialized")
    
    async def start_monitoring(self, use_simulation: bool = True):
        """
        Start real-time monitoring.
        
        Args:
            use_simulation: If True, use simulated traffic; if False, wait for real traffic
        """
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        if not self.ml_service.models_loaded:
            logger.error("Cannot start monitoring: ML models not loaded")
            raise RuntimeError("ML models not loaded")
        
        self.is_running = True
        logger.info("Starting real-time monitoring...")
        
        if use_simulation:
            self.monitoring_task = asyncio.create_task(self._simulation_loop())
        else:
            self.monitoring_task = asyncio.create_task(self._real_traffic_loop())
    
    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time monitoring...")
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time monitoring stopped")
    
    async def _simulation_loop(self):
        """
        Simulation loop: generates synthetic traffic and processes it.
        This simulates real-time network traffic for testing.
        """
        logger.info("Starting traffic simulation loop...")
        
        try:
            while self.is_running:
                # Generate synthetic network traffic
                synthetic_data = self._generate_synthetic_traffic(self.traffic_batch_size)
                
                # Process through ML models
                await self._process_traffic_batch(synthetic_data)
                
                # Wait before next batch
                await asyncio.sleep(self.traffic_interval)
                
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.is_running = False
    
    async def _real_traffic_loop(self):
        """
        Real traffic loop: processes actual network traffic.
        This would integrate with actual network capture tools.
        """
        logger.info("Starting real traffic monitoring loop...")
        
        # TODO: Integrate with actual packet capture (e.g., scapy, pyshark)
        # For now, this is a placeholder
        
        try:
            while self.is_running:
                # Wait for real traffic
                # In production, this would capture actual packets
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Real traffic loop cancelled")
        except Exception as e:
            logger.error(f"Error in real traffic loop: {e}")
            self.is_running = False
    
    def _generate_synthetic_traffic(self, count: int) -> pd.DataFrame:
        """
        Generate synthetic network traffic for simulation.
        Creates realistic-looking network features.
        
        Args:
            count: Number of traffic samples to generate
            
        Returns:
            DataFrame with synthetic network features
        """
        # Determine number of features expected by model
        n_features = self.ml_service.expected_features or 41
        
        # Generate random traffic with various patterns
        data = {}
        
        # Mix of normal and attack traffic (70% normal, 30% attack)
        is_attack = np.random.random(count) > 0.7
        
        for i in range(n_features):
            # Generate features with different characteristics
            if is_attack.any():
                # Attack traffic has different patterns
                normal_values = np.random.exponential(scale=100, size=count)
                attack_values = np.random.exponential(scale=500, size=count)
                
                # Mix based on is_attack flag
                feature_values = np.where(is_attack, attack_values, normal_values)
            else:
                feature_values = np.random.exponential(scale=100, size=count)
            
            data[f'feature_{i}'] = feature_values
        
        return pd.DataFrame(data)
    
    async def _process_traffic_batch(self, traffic_df: pd.DataFrame):
        """
        Process a batch of network traffic through ML models.
        
        Args:
            traffic_df: DataFrame containing network traffic features
        """
        try:
            # Validate and prepare data
            X, feature_names = self.ml_service.validate_and_prepare_data(traffic_df)
            
            # Make predictions
            predictions = self.ml_service.predict(X)
            
            # Extract predictions
            binary_preds = predictions["binary_pred"]
            multiclass_preds = predictions["multiclass_pred"]
            
            # Get confidence scores
            if "binary_proba" in predictions:
                confidences = predictions["binary_proba"].max(axis=1)
            else:
                confidences = np.ones(len(binary_preds)) * 0.95
            
            # Update metrics
            self.metrics.add_batch_predictions(binary_preds, multiclass_preds, confidences)
            
            # Log intrusions
            for i in range(len(binary_preds)):
                if binary_preds[i] == 1:  # Attack detected
                    self._log_intrusion(
                        multiclass_pred=int(multiclass_preds[i]),
                        confidence=float(confidences[i]),
                        features=X[i] if i < len(X) else None
                    )
            
            logger.debug(f"Processed batch: {len(binary_preds)} samples, "
                        f"{np.sum(binary_preds)} attacks detected")
            
        except Exception as e:
            logger.error(f"Error processing traffic batch: {e}")
    
    def _log_intrusion(self, multiclass_pred: int, confidence: float, 
                      features: Optional[np.ndarray] = None):
        """
        Log detected intrusion to network logger.
        
        Args:
            multiclass_pred: Multi-class prediction (attack type)
            confidence: Prediction confidence
            features: Optional feature vector
        """
        from app.config import MULTICLASS_LABELS
        
        attack_type = MULTICLASS_LABELS.get(multiclass_pred, "Unknown")
        
        # Generate random IPs for simulation
        source_ip = f"{np.random.randint(1,255)}.{np.random.randint(0,255)}." \
                   f"{np.random.randint(0,255)}.{np.random.randint(1,254)}"
        dest_ip = "192.168.1.100"  # Internal server
        
        self.network_logger.log_intrusion(
            source_ip=source_ip,
            destination_ip=dest_ip,
            intrusion_type=attack_type,
            confidence=confidence,
            action="logged",
            model_prediction=attack_type,
            features={"confidence": confidence}
        )
    
    async def process_uploaded_data(self, df: pd.DataFrame):
        """
        Process uploaded CSV data in real-time mode.
        Simulates real-time processing of historical data.
        
        Args:
            df: DataFrame with network traffic data
        """
        logger.info(f"Processing uploaded data in real-time mode: {len(df)} samples")
        
        # Process in batches to simulate real-time
        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            await self._process_traffic_batch(batch)
            
            # Small delay to simulate real-time processing
            if self.is_running:
                await asyncio.sleep(0.1)
        
        logger.info("Finished processing uploaded data")
    
    def get_metrics(self) -> Dict:
        """Get current real-time metrics."""
        return self.metrics.get_current_stats()
    
    def get_timeline(self, num_points: int = 50) -> Dict:
        """Get timeline data for visualization."""
        return self.metrics.get_timeline_data(num_points)
    
    def get_confidence_distribution(self) -> Dict:
        """Get confidence distribution for histogram."""
        return self.metrics.get_confidence_distribution()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.reset()
        logger.info("Metrics reset")


# Global real-time monitoring service instance
# Will be initialized in main.py with network_logger
realtime_service: Optional[RealTimeMonitoringService] = None


def initialize_realtime_service(network_logger: NetworkLogger):
    """
    Initialize the global real-time service instance.
    
    Args:
        network_logger: NetworkLogger instance
    """
    global realtime_service
    realtime_service = RealTimeMonitoringService(network_logger)
    logger.info("Real-time service initialized")
    return realtime_service
