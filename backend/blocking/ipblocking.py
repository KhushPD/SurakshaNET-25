"""IP Blocking Module"""
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import logging

logger = logging.getLogger(__name__)

class IPBlockingService:
    """Manages IP blocking for detected threats"""
    
    def __init__(self):
        self.blocked_ips: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.recent_blocks: List[Dict] = []  # Store recent blocks for notifications
    
    def block_ip(self, ip: str, reason: str, duration_minutes: int = 60, confidence: float = None) -> None:
        """Block an IP address for specified duration"""
        now = datetime.now()
        with self.lock:
            self.blocked_ips[ip] = {
                "blocked_at": now,
                "reason": reason,
                "expires": now + timedelta(minutes=duration_minutes),
                "status": "blocked"
            }
            # Add to recent blocks for notification
            self.recent_blocks.append({
                "ip": ip,
                "reason": reason,
                "timestamp": now.isoformat(),
                "confidence": f"{confidence*100:.1f}%" if confidence else "Manual",
                "id": f"{ip}_{int(now.timestamp())}"
            })
            # Keep only last 50 blocks
            if len(self.recent_blocks) > 3:
                self.recent_blocks.pop(0)
        
        logger.warning(f"🚫 Blocked IP: {ip} | Reason: {reason} | Duration: {duration_minutes}m")
        
        # Send email alert
        try:
            from reporting.mail_alert import email_alert_service
            email_alert_service.send_ip_block_alert(ip, reason, confidence)
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        with self.lock:
            if ip in self.blocked_ips:
                if datetime.now() < self.blocked_ips[ip]["expires"]:
                    return True
                else:
                    del self.blocked_ips[ip]
        return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP"""
        with self.lock:
            if ip in self.blocked_ips:
                del self.blocked_ips[ip]
                logger.info(f"✓ Unblocked IP: {ip}")
                return True
        return False
    
    def get_blocked_ips(self) -> List[Dict]:
        """Get list of currently blocked IPs"""
        with self.lock:
            now = datetime.now()
            blocked = []
            for ip, data in list(self.blocked_ips.items()):
                if now < data["expires"]:
                    blocked.append({
                        "ip": ip,
                        "reason": data["reason"],
                        "blocked_at": data["blocked_at"].isoformat(),
                        "expires": data["expires"].isoformat()
                    })
                else:
                    del self.blocked_ips[ip]
            return blocked
    
    def get_recent_blocks(self, since_timestamp: str = None) -> List[Dict]:
        """Get recent IP blocks for notifications"""
        with self.lock:
            if since_timestamp:
                return [block for block in self.recent_blocks if block["timestamp"] > since_timestamp]
            return self.recent_blocks.copy()

# Global instance
ip_blocking_service = IPBlockingService()