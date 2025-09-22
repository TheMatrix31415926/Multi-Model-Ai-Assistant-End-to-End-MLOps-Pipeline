# monitoring/dashboard_generator.py - Generate monitoring dashboard
from datetime import datetime, timedelta
from typing import Dict, Any
import json
from pathlib import Path
from multimodal_ai_assistant.logger import get_logger  # adjust if logger is in a different path


class MonitoringDashboard:
    """Generate monitoring dashboard data"""

    def __init__(self, health_monitor, log_analyzer):
        self.health_monitor = health_monitor
        self.log_analyzer = log_analyzer

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        current_time = datetime.now()

        # Get current health status
        health_status = self.health_monitor.get_health_status()

        # Get recent log analysis
        log_analysis = self.log_analyzer.analyze_recent_logs(minutes=15)

        # Get error summary
        error_summary = self.log_analyzer.get_error_summary(hours=24)

        # Calculate uptime (mock - in real implementation, track actual uptime)
        uptime = "99.9%"  # Placeholder for real uptime calculation

        return {
            "timestamp": current_time.isoformat(),
            "summary": {
                "overall_status": health_status.get("overall_status", "unknown"),
                "uptime": uptime,
                "total_services": len(health_status.get("services", {})),
                "healthy_services": len([
                    s for s in health_status.get("services", {}).values()
                    if s.get("status") == "healthy"
                ]),
                "total_errors_24h": error_summary.get("total_errors", 0),
                "alerts_last_15min": log_analysis.get("alerts_generated", 0)
            },
            "services": health_status.get("services", {}),
            "system_resources": health_status.get("system", {}),
            "recent_logs": log_analysis,
            "error_trends": error_summary,
            "quick_actions": [
                {"name": "Restart API", "command": "docker-compose restart api"},
                {"name": "View API Logs", "command": "docker-compose logs api"},
                {"name": "Check Disk Space", "command": "df -h"},
                {"name": "View System Load", "command": "top"}
            ]
        }

    def save_dashboard_data(self, filename: str = None):
        """Save dashboard data to file"""
        if filename is None:
            filename = f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        dashboard_data = self.generate_dashboard_data()

        dashboard_file = Path("logs") / filename
        dashboard_file.parent.mkdir(exist_ok=True)

        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        return dashboard_file


# ======================
# Decorators
# ======================

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
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'error',
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    return wrapper


def log_api_request(func):
    """Decorator to log API requests"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        request = kwargs.get('request') or (args[0] if args else None)

        request_id = getattr(getattr(request, 'state', None), 'request_id', 'unknown')

        logger.info(
            f"API request: {func.__name__}",
            extra={
                'request_id': request_id,
                'endpoint': func.__name__,
                'method': getattr(request, 'method', 'unknown'),
                'status': 'started'
            }
        )

        try:
            result = func(*args, **kwargs)
            logger.info(
                f"API request completed: {func.__name__}",
                extra={
                    'request_id': request_id,
                    'endpoint': func.__name__,
                    'status': 'completed'
                }
            )
            return result
        except Exception as e:
            logger.error(
                f"API request failed: {func.__name__}",
                extra={
                    'request_id': request_id,
                    'endpoint': func.__name__,
                    'status': 'failed',
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    return wrapper
