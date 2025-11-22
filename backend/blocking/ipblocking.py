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
    
    def block_ip(self, ip: str, reason: str, duration_minutes: int = 60) -> None:
        """Block an IP address for specified duration"""
        with self.lock:
            self.blocked_ips[ip] = {
                "blocked_at": datetime.now(),
                "reason": reason,
                "expires": datetime.now() + timedelta(minutes=duration_minutes),
                "status": "blocked"
            }
        logger.warning(f"🚫 Blocked IP: {ip} | Reason: {reason} | Duration: {duration_minutes}m")
    
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

# Global instance
ip_blocking_service = IPBlockingService()