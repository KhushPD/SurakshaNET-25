"""
Email Alert System for IP Blocking Events
==========================================
Sends email notifications when IPs are blocked due to threats
Implements batched notifications to prevent spam during high-frequency attacks
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from typing import List, Dict
import threading

load_dotenv()
logger = logging.getLogger(__name__)

class EmailAlertService:
    """Service for sending email alerts on security events with batching"""
    
    def __init__(self, throttle_seconds: int = 10):
        self.sender_email = os.getenv('ALERT_EMAIL', 'vatsalpjain@gmail.com')
        self.receiver_email = os.getenv('OPERATOR_EMAIL', 'dhoonpandya@gmail.com')
        self.password = os.getenv('EMAIL_PASSWORD', 'Dhoon2212')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
        # Batching configuration
        self.throttle_seconds = throttle_seconds
        self.last_email_time = None
        self.pending_blocks: List[Dict] = []
        self.lock = threading.Lock()
        
    def send_ip_block_alert(self, ip: str, reason: str, confidence: float = None) -> bool:
        """Queue IP block for batched notification"""
        with self.lock:
            self.pending_blocks.append({
                'ip': ip,
                'reason': reason,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Check if we should send email now
            now = datetime.now()
            should_send = (
                self.last_email_time is None or 
                (now - self.last_email_time).total_seconds() >= self.throttle_seconds
            )
            
            if should_send and self.pending_blocks:
                return self._send_batched_alert()
            else:
                logger.info(f"📧 Queued notification for {ip} (batch size: {len(self.pending_blocks)})")
                return True
    
    def _send_batched_alert(self) -> bool:
        """Send batched email alert for multiple IP blocks"""
        try:
            if not self.pending_blocks:
                return True
                
            block_count = len(self.pending_blocks)
            
            # Prepare subject
            if block_count == 1:
                subject = f"🚨 Security Alert: IP {self.pending_blocks[0]['ip']} Blocked"
            else:
                subject = f"🚨 Security Alert: {block_count} IPs Blocked"
            
            # Prepare body
            body_lines = [
                "Security Alert - IP Address(es) Blocked",
                "=" * 60,
                f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total IPs Blocked: {block_count}\n"
            ]
            
            # Add details for each blocked IP
            for i, block in enumerate(self.pending_blocks[:20], 1):  # Limit to 20 in email
                conf_str = f"{block['confidence']*100:.1f}%" if block['confidence'] else "Manual"
                body_lines.append(f"{i}. IP: {block['ip']}")
                body_lines.append(f"   Reason: {block['reason']}")
                body_lines.append(f"   Confidence: {conf_str}")
                body_lines.append(f"   Time: {block['timestamp'].strftime('%H:%M:%S')}\n")
            
            if block_count > 20:
                body_lines.append(f"... and {block_count - 20} more IPs\n")
            
            body_lines.extend([
                "\nAction Taken: All IPs have been automatically blocked from accessing the network.",
                "\nPlease review the logs and take necessary action if needed.",
                "\n---",
                "SurakshaNET-25 Security System"
            ])
            
            body = "\n".join(body_lines)
            
            # Send email
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, self.receiver_email, message.as_string())
            
            logger.info(f"✓ Batched email alert sent for {block_count} blocked IP(s)")
            
            # Clear pending blocks and update timestamp
            self.pending_blocks.clear()
            self.last_email_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send batched email alert: {e}")
            return False
    
    def flush_pending(self) -> bool:
        """Force send any pending notifications"""
        with self.lock:
            if self.pending_blocks:
                return self._send_batched_alert()
            return True

# Global instance
email_alert_service = EmailAlertService(throttle_seconds=10)

