# Add to your FastAPI backend (fastapi_backend.py)

from fastapi import Request, HTTPException
from datetime import datetime, timedelta

# Blocked IPs storage
blocked_ips = {}  # {ip: {"blocked_at": datetime, "reason": str, "expires": datetime}}

def block_ip(ip: str, reason: str, duration_minutes: int = 60):
    """Block an IP address for a specified duration"""
    blocked_ips[ip] = {
        "blocked_at": datetime.now(),
        "reason": reason,
        "expires": datetime.now() + timedelta(minutes=duration_minutes),
        "status": "blocked"
    }
    print(f"🚫 Blocked IP: {ip} | Reason: {reason} | Duration: {duration_minutes}m")

def is_ip_blocked(ip: str) -> bool:
    """Check if an IP is currently blocked"""
    if ip in blocked_ips:
        if datetime.now() < blocked_ips[ip]["expires"]:
            return True
        else:
            # Expired, remove from blocked list
            del blocked_ips[ip]
    return False

# Middleware to check blocked IPs
@app.middleware("http")
async def block_malicious_ips(request: Request, call_next):
    client_ip = request.client.host
    
    if is_ip_blocked(client_ip):
        raise HTTPException(
            status_code=403, 
            detail=f"IP {client_ip} is blocked due to suspicious activity"
        )
    
    response = await call_next(request)
    return response