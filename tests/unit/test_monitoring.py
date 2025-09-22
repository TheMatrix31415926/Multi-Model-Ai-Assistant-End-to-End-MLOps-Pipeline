# tests/unit/test_monitoring.py - Monitoring component tests
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime
from monitoring.health_checker import HealthMonitor, HealthCheck
from monitoring.alerting.alert_manager import AlertManager, Alert
from monitoring.log_analyzer import LogAnalyzer

class TestHealthMonitor:
    """Test health monitoring functionality"""
    
    def test_health_check_creation(self):
        """Test health check creation"""
        check = HealthCheck(
            name="Test Service",
            check_type="http",
            target="http://localhost:8000/health"
        )
        
        assert check.name == "Test Service"
        assert check.check_type == "http"
        assert check.timeout == 10
        assert check.interval == 30
    
    @pytest.mark.asyncio
    async def test_check_http_endpoint_success(self):
        """Test successful HTTP health check"""
        mock_alert_manager = Mock()
        monitor = HealthMonitor(mock_alert_manager)
        
        check = HealthCheck("API", "http", "http://localhost:8000/health")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.__aenter__ = asyncio.coroutine(lambda self: self)
            mock_response.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.__aenter__ = asyncio.coroutine(lambda self: self)
            mock_session_instance.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            mock_session.return_value = mock_session_instance
            
            result = await monitor.check_http_endpoint(check)
            
            assert result["status"] == "healthy"
            assert result["status_code"] == 200
            assert "response_time" in result
    
    @pytest.mark.asyncio  
    async def test_check_http_endpoint_failure(self):
        """Test failed HTTP health check"""
        mock_alert_manager = Mock()
        monitor = HealthMonitor(mock_alert_manager)
        
        check = HealthCheck("API", "http", "http://localhost:8000/health")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.__aenter__ = asyncio.coroutine(lambda self: self)
            mock_response.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.__aenter__ = asyncio.coroutine(lambda self: self)
            mock_session_instance.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            mock_session.return_value = mock_session_instance
            
            result = await monitor.check_http_endpoint(check)
            
            assert result["status"] == "unhealthy"
            assert result["status_code"] == 500

class TestAlertManager:
    """Test alert management functionality"""
    
    def test_alert_creation(self):
        """Test alert object creation"""
        alert = Alert(
            name="Test Alert",
            severity="warning",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test"
        )
        
        assert alert.name == "Test Alert"
        assert alert.severity == "warning"
        assert alert.source == "test"
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization"""
        config = {
            'routing': {
                'critical': {'email_recipients': ['admin@test.com']},
                'warning': {'email_recipients': ['team@test.com']}
            }
        }
        
        manager = AlertManager(config)
        assert manager.config == config
        assert manager.active_alerts == {}
    
    @patch('smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp):
        """Test email alert sending"""
        config = {
            'smtp': {
                'host': 'localhost',
                'port': 587,
                'from_email': 'alerts@test.com'
            }
        }
        
        manager = AlertManager(config)
        alert = Alert(
            name="Test Alert",
            severity="critical",
            message="Test message",
            timestamp=datetime.now(),
            source="test"
        )
        
        # Mock SMTP
        mock_server = Mock()
        mock_smtp.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = Mock(return_value=None)
        
        manager.send_email_alert(alert, ['test@example.com'])
        
        # Verify SMTP was called
        mock_smtp.assert_called_once()

class TestLogAnalyzer:
    """Test log analysis functionality"""
    
    def test_log_pattern_matching(self):
        """Test log pattern detection"""
        mock_alert_manager = Mock()
        analyzer = LogAnalyzer(mock_alert_manager)
        
        # Test error pattern
        assert analyzer.log_patterns['error'].search('ERROR: Something went wrong')
        assert analyzer.log_patterns['warning'].search('WARNING: Low disk space')
        assert analyzer.log_patterns['timeout'].search('Request timed out')
    
    def test_parse_log_line_json(self):
        """Test JSON log line parsing"""
        mock_alert_manager = Mock()
        analyzer = LogAnalyzer(mock_alert_manager)
        
        json_line = '{"timestamp": "2024-01-01T00:00:00", "level": "INFO", "message": "Test"}'
        result = analyzer.parse_log_line(json_line)
        
        assert result['timestamp'] == "2024-01-01T00:00:00"
        assert result['level'] == "INFO"
        assert result['message'] == "Test"
    
    def test_parse_log_line_plain_text(self):
        """Test plain text log line parsing"""
        mock_alert_manager = Mock()
        analyzer = LogAnalyzer(mock_alert_manager)
        
        plain_line = "2024-01-01 ERROR: Something went wrong"
        result = analyzer.parse_log_line(plain_line)
        
        assert result['message'] == plain_line
        assert result['raw'] == True