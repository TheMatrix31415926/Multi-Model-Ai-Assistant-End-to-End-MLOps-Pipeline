# monitoring/alerting/alert_manager.py - Custom alert management
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None

class AlertManager:
    """Custom alert management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_alerts = {}
        
    def send_email_alert(self, alert: Alert, recipients: List[str]):
        """Send email alert"""
        try:
            smtp_config = self.config.get('smtp', {})
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email', 'alerts@multimodal-ai.com')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.name}"
            
            # Email body
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity}
            Timestamp: {alert.timestamp.isoformat()}
            Source: {alert.source}
            
            Message: {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata or {}, indent=2)}
            
            ---
            Multi-Modal AI Assistant Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config.get('host', 'localhost'), smtp_config.get('port', 587)) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent: {alert.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: Alert, webhook_url: str):
        """Send Slack alert"""
        try:
            color_map = {
                'critical':'#FF0000',
                'warning': '#FFA500',
                'info': '#00FF00'
            }
            
            payload = {
                "text": f"Multi-Modal AI Alert: {alert.name}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity.lower(), '#808080'),
                        "fields": [
                            {"title": "Severity", "value": alert.severity, "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True},
                            {"title": "Message", "value": alert.message, "short": False}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent: {alert.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")