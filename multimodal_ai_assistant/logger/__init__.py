"""Package initialization file"""
# multimodal_ai_assistant/logger/__init__.py - Centralized logging setup
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import traceback
import requests
from dataclasses import dataclass, field
from datetime import datetime
import time
from fastapi import Request



class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'conversation_id'):
            log_entry['conversation_id'] = record.conversation_id
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

class MultiModalLogger:
    """Centralized logger for Multi-Modal AI Assistant"""
    
    def __init__(self, name: str = "multimodal_ai", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        self._setup_error_handler()
    
    def _setup_console_handler(self):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log levels"""
        
        # All logs (rotating file handler)
        all_logs_handler = logging.handlers.RotatingFileHandler(
            'logs/multimodal_ai.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        all_logs_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(all_logs_handler)
        
        # Application logs (INFO and above)
        app_logs_handler = logging.handlers.RotatingFileHandler(
            'logs/app.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        app_logs_handler.setLevel(logging.INFO)
        app_logs_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(app_logs_handler)
        
        # API access logs
        access_logs_handler = logging.handlers.RotatingFileHandler(
            'logs/access.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        access_logs_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(access_logs_handler)
    
    def _setup_error_handler(self):
        """Setup error handler for ERROR and CRITICAL logs"""
        error_handler = logging.handlers.RotatingFileHandler(
            'logs/errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)
    
    def get_logger(self):
        return self.logger

# Global logger instance
_logger_instance = None

def get_logger(name: str = "multimodal_ai") -> logging.Logger:
    """Get or create logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MultiModalLogger(name)
    return _logger_instance.get_logger()



# Logging decorator
def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Function {func.__name__} executed successfully",
                extra={
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'success'
                }
            )
            return result
        except Exception as e:
            logger.error(
                f"Function {func.__name__} failed: {e}",
                exc_info=True,
                extra={'function': func.__name__, 'status': 'error'}
            )
            raise
    return wrapper


# Example Alert class
class Alert:
    def __init__(self, name, severity, message, timestamp, source, metadata=None):
        self.name = name
        self.severity = severity
        self.message = message
        self.timestamp = timestamp
        self.source = source
        self.metadata = metadata or {}


# Alert manager
class AlertManager:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.active_alerts = {}

    @log_execution_time
    def send_webhook_alert(self, alert: Alert, webhook_url: str):
        """Send webhook alert"""
        try:
            payload = {
                "alert": {
                    "name": alert.name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source,
                    "metadata": alert.metadata or {}
                }
            }

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            self.logger.info(f"Webhook alert sent: {alert.name}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    @log_execution_time
    def process_alert(self, alert: Alert):
        """Process and route alert based on configuration"""
        alert_key = f"{alert.name}_{alert.source}"

        if alert_key in self.active_alerts:
            last_sent = self.active_alerts[alert_key]
            time_diff = (alert.timestamp - last_sent).total_seconds()
            if time_diff < 300:  # avoid spam
                return

        self.active_alerts[alert_key] = alert.timestamp

        routing_config = self.config.get('routing', {})
        severity_config = routing_config.get(alert.severity.lower(), {})

        # Email alerts
        email_recipients = severity_config.get('email_recipients', [])
        if email_recipients:
            self.send_email_alert(alert, email_recipients)

        # Slack alerts
        slack_webhook = severity_config.get('slack_webhook')
        if slack_webhook:
            self.send_slack_alert(alert, slack_webhook)

        # Webhook alerts
        webhook_url = severity_config.get('webhook_url')
        if webhook_url:
            self.send_webhook_alert(alert, webhook_url)

        self.logger.info(f"Alert processed: {alert.name} ({alert.severity})")

    @log_execution_time
    def create_system_alert(self, metric_name: str, current_value: float, threshold: float, operator: str = '>'):
        """Create system performance alert"""
        if operator == '>' and current_value > threshold:
            severity = 'critical' if current_value > threshold * 1.2 else 'warning'

            alert = Alert(
                name=f"High {metric_name}",
                severity=severity,
                message=f"{metric_name} is {current_value:.1f}%, threshold is {threshold}%",
                timestamp=datetime.now(),
                source="system_monitor",
                metadata={
                    "metric": metric_name,
                    "current_value": current_value,
                    "threshold": threshold,
                    "operator": operator
                }
            )

            self.process_alert(alert)

    @log_execution_time
    def create_api_alert(self, error_type: str, error_count: int, endpoint: str):
        """Create API error alert"""
        severity = 'critical' if error_count > 10 else 'warning'

        alert = Alert(
            name=f"API {error_type} Errors",
            severity=severity,
            message=f"{error_count} {error_type} errors detected on {endpoint}",
            timestamp=datetime.now(),
            source="api_monitor",
            metadata={
                "error_type": error_type,
                "error_count": error_count,
                "endpoint": endpoint
            }
        )

        self.process_alert(alert)