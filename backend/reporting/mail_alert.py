"""
Email Alert System for IP Blocking Events
==========================================
Sends email notifications when IPs are blocked due to threats
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EmailAlertService:
    """Service for sending email alerts on security events"""
    
    def __init__(self):
        self.sender_email = os.getenv('ALERT_EMAIL', 'vatsalpjain@gmail.com')
        self.receiver_email = os.getenv('OPERATOR_EMAIL', 'dhoonpandya@gmail.com')
        self.password = os.getenv('EMAIL_PASSWORD', 'Dhoon2212')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
    def send_ip_block_alert(self, ip: str, reason: str, confidence: float = None) -> bool:
        """Send email alert when IP is blocked"""
        try:
            subject = f"🚨 Security Alert: IP {ip} Blocked"
            
            body = f"""
Security Alert - IP Address Blocked
====================================

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Blocked IP: {ip}
Reason: {reason}
{f'Confidence: {confidence*100:.1f}%' if confidence else ''}

Action Taken: IP has been automatically blocked from accessing the network.

Please review the logs and take necessary action if needed.

---
SurakshaNET-25 Security System
"""
            
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, self.receiver_email, message.as_string())
            
            logger.info(f"✓ Email alert sent for blocked IP: {ip}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")
            return False

# Global instance
email_alert_service = EmailAlertService()

