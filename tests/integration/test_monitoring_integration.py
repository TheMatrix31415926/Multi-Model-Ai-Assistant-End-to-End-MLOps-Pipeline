# tests/integration/test_monitoring_integration.py - Monitoring integration tests
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import os

from monitoring.health_checker import HealthMonitor
from monitoring.alerting.alert_manager import AlertManager, Alert
from monitoring.log_analyzer import LogAnalyzer
from monitoring.dashboard_generator import MonitoringDashboard

class TestMonitoringIntegration:
    """Test monitoring system integration"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        # Create logs directory
        os.makedirs(os.path.join(self.temp_dir, "logs"), exist_ok=True)
        
        # Change to temp directory for tests
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        # Cleanup
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_log_file(self, filename, entries):
        """Create test log file with entries"""
        log_path = os.path.join("logs", filename)
        with open(log_path, 'w') as f:
            for entry in entries:
                if isinstance(entry, dict):
                    f.write(json.dumps(entry) + "\n")
                else:
                    f.write(str(entry) + "\n")
    
    def test_alert_manager_integration(self):
        """Test alert manager with different alert types"""
        
        # Mock configuration
        config = {
            'routing': {
                'critical': {
                    'email_recipients': ['admin@test.com'],
                    'webhook_url': 'http://localhost:8000/webhooks/test'
                },
                'warning': {
                    'email_recipients': ['team@test.com']
                }
            },
            'smtp': {
                'host': 'localhost',
                'port': 587,
                'from_email': 'alerts@test.com'
            }
        }
        
        alert_manager = AlertManager(config)
        
        # Test critical alert
        critical_alert = Alert(
            name="Service Down",
            severity="critical",
            message="API service is not responding",
            timestamp=datetime.now(),
            source="health_monitor"
        )
        
        with patch.object(alert_manager, 'send_email_alert') as mock_email, \
             patch.object(alert_manager, 'send_webhook_alert') as mock_webhook:
            
            alert_manager.process_alert(critical_alert)
            
            # Verify critical alert triggers email and webhook
            mock_email.assert_called_once()
            mock_webhook.assert_called_once()
        
        # Test warning alert  
        warning_alert = Alert(
            name="High CPU",
            severity="warning",
            message="CPU usage is 85%",
            timestamp=datetime.now(),
            source="system_monitor"
        )
        
        with patch.object(alert_manager, 'send_email_alert') as mock_email:
            alert_manager.process_alert(warning_alert)
            mock_email.assert_called_once()
        
        print(" Alert manager integration test passed!")
    
    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test health monitor with multiple services"""
        
        mock_alert_manager = Mock()
        health_monitor = HealthMonitor(mock_alert_manager)
        
        # Mock HTTP responses for different services
        with patch('aiohttp.ClientSession') as mock_session_class:
            
            # Setup different responses for different endpoints
            def mock_get_response(url):
                mock_response = Mock()
                mock_response.__aenter__ = asyncio.coroutine(lambda self: self)
                mock_response.__aexit__ = asyncio.coroutine(lambda self, *args: None)
                
                if "health" in str(url):
                    mock_response.status = 200  # API healthy
                elif "9090" in str(url):
                    mock_response.status = 200  # Prometheus healthy  
                elif "3000" in str(url):
                    mock_response.status = 503  # Grafana unhealthy
                else:
                    mock_response.status = 200  # Default healthy
                
                return mock_response
            
            mock_session = Mock()
            mock_session.get.side_effect = mock_get_response
            mock_session.__aenter__ = asyncio.coroutine(lambda self: self)
            mock_session.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            mock_session_class.return_value = mock_session
            
            # Run health checks
            results = await health_monitor.run_health_checks()
            
            # Verify results
            assert "API Health" in results
            assert results["API Health"]["status"] == "healthy"
            
            # Check that unhealthy service triggered alert
            mock_alert_manager.process_alert.assert_called()
        
        print(" Health monitor integration test passed!")
    
    def test_log_analyzer_integration(self):
        """Test log analyzer with real log files"""
        
        mock_alert_manager = Mock()
        log_analyzer = LogAnalyzer(mock_alert_manager)
        
        # Create test log entries
        log_entries = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Application started successfully"
            },
            {
                "timestamp": datetime.now().isoformat(), 
                "level": "ERROR",
                "message": "Database connection failed"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR", 
                "message": "HTTP request timeout occurred"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "WARNING",
                "message": "High memory usage detected"
            }
        ]
        
        # Create log file
        self.create_test_log_file("multimodal_ai.log", log_entries)
        
        # Analyze recent logs
        analysis = log_analyzer.analyze_recent_logs(minutes=60)
        
        # Verify analysis results
        assert analysis["total_entries"] > 0
        assert "pattern_counts" in analysis
        assert analysis["pattern_counts"].get("error", 0) >= 2
        assert analysis["alerts_generated"] >= 0
        
        print(" Log analyzer integration test passed!")
    
    def test_dashboard_generator_integration(self):
        """Test dashboard generator with all components"""
        
        # Setup mock components
        mock_alert_manager = Mock()
        mock_health_monitor = Mock()
        mock_log_analyzer = Mock()
        
        # Setup mock return values
        mock_health_monitor.get_health_status.return_value = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {
                "api": {"status": "healthy", "response_time": 0.1},
                "database": {"status": "healthy", "response_time": 0.05}
            },
            "system": {
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "disk_usage": 75.0
            }
        }
        
        mock_log_analyzer.analyze_recent_logs.return_value = {
            "total_entries": 100,
            "pattern_counts": {"error": 2, "warning": 5},
            "alerts_generated": 1
        }
        
        mock_log_analyzer.get_error_summary.return_value = {
            "total_errors": 10,
            "period": "Last 24 hours",
            "error_types": {"timeout": 5, "connection": 3, "other": 2}
        }
        
        # Create dashboard generator
        dashboard = MonitoringDashboard(mock_health_monitor, mock_log_analyzer)
        
        # Generate dashboard data
        dashboard_data = dashboard.generate_dashboard_data()
        
        # Verify dashboard structure
        assert "timestamp" in dashboard_data
        assert "summary" in dashboard_data
        assert dashboard_data["summary"]["overall_status"] == "healthy"
        assert "services" in dashboard_data
        assert "system_resources" in dashboard_data
        assert "recent_logs" in dashboard_data
        assert "error_trends" in dashboard_data
        
        # Test saving dashboard data
        saved_file = dashboard.save_dashboard_data("test_dashboard.json")
        assert os.path.exists(saved_file)
        
        # Verify saved content
        with open(saved_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["summary"]["overall_status"] == "healthy"
        
        print(" Dashboard generator integration test passed!")