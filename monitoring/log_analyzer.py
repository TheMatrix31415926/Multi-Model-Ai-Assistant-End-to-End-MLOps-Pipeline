# monitoring/log_analyzer.py - Log analysis and alerting
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Iterator
import logging
from pathlib import Path
from multimodal_ai_assistant.logger import get_logger
from monitoring.alerting.alert_manager import AlertManager, Alert

class LogAnalyzer:
    """Analyze logs for patterns and anomalies"""
    
    def __init__(self, alert_manager: AlertManager):
        self.logger = get_logger()
        self.alert_manager = alert_manager
        self.log_patterns = {
            'error': re.compile(r'ERROR|CRITICAL|Exception|Traceback', re.IGNORECASE),
            'warning': re.compile(r'WARNING|WARN', re.IGNORECASE),
            'timeout': re.compile(r'timeout|timed out', re.IGNORECASE),
            'connection_error': re.compile(r'connection.*error|connection.*refused|connection.*failed', re.IGNORECASE),
            'memory_error': re.compile(r'out of memory|memory error|malloc failed', re.IGNORECASE),
            'auth_failure': re.compile(r'authentication.*failed|unauthorized|access.*denied', re.IGNORECASE)
        }
        
        self.error_thresholds = {
            'error': 10,  # Alert if more than 10 errors in 5 minutes
            'warning': 50,  # Alert if more than 50 warnings in 5 minutes
            'timeout': 5,   # Alert if more than 5 timeouts in 5 minutes
            'connection_error': 5,
            'memory_error': 1,  # Alert on any memory error
            'auth_failure': 10
        }
    
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single log line"""
        try:
            # Try to parse as JSON first
            return json.loads(line.strip())
        except json.JSONDecodeError:
            # Fallback to simple parsing
            return {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': line.strip(),
                'raw': True
            }
    
    def read_log_file(self, log_file: Path, since: datetime = None) -> Iterator[Dict[str, Any]]:
        """Read log file and yield parsed entries"""
        if not log_file.exists():
            return
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    
                    # Filter by timestamp if specified
                    if since:
                        try:
                            entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                            if entry_time < since:
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    yield entry
                    
        except Exception as e:
            self.logger.error(f"Error reading log file {log_file}: {e}")
    
    def analyze_patterns(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze log entries for patterns"""
        pattern_counts = defaultdict(int)
        error_details = defaultdict(list)
        level_counts = Counter()
        
        for entry in log_entries:
            message = entry.get('message', '').lower()
            level = entry.get('level', 'INFO').upper()
            timestamp = entry.get('timestamp', '')
            
            level_counts[level] += 1
            
            # Check for patterns
            for pattern_name, pattern_regex in self.log_patterns.items():
                if pattern_regex.search(message):
                    pattern_counts[pattern_name] += 1
                    error_details[pattern_name].append({
                        'timestamp': timestamp,
                        'message': entry.get('message', ''),
                        'level': level
                    })
        
        return {
            'pattern_counts': dict(pattern_counts),
            'error_details': dict(error_details),
            'level_counts': dict(level_counts),
            'total_entries': len(log_entries)
        }
    
    def check_anomalies(self, analysis: Dict[str, Any]) -> List[Alert]:
        """Check for anomalies and create alerts"""
        alerts = []
        pattern_counts = analysis['pattern_counts']
        error_details = analysis['error_details']
        
        for pattern_name, count in pattern_counts.items():
            threshold = self.error_thresholds.get(pattern_name, 10)
            
            if count >= threshold:
                # Get recent examples
                recent_examples = error_details.get(pattern_name, [])[-3:]
                
                alert = Alert(
                    name=f"High {pattern_name.replace('_', ' ').title()} Rate",
                    severity="critical" if pattern_name in ['error', 'memory_error'] else "warning",
                    message=f"Detected {count} {pattern_name} events (threshold: {threshold})",
                    timestamp=datetime.now(),
                    source="log_analyzer",
                    metadata={
                        "pattern": pattern_name,
                        "count": count,
                        "threshold": threshold,
                        "recent_examples": recent_examples
                    }
                )
                
                alerts.append(alert)
        
        return alerts
    
    def analyze_recent_logs(self, minutes: int = 5) -> Dict[str, Any]:
        """Analyze logs from the last N minutes"""
        since = datetime.now() - timedelta(minutes=minutes)
        log_files = [
            Path("logs/multimodal_ai.log"),
            Path("logs/app.log"),
            Path("logs/errors.log")
        ]
        
        all_entries = []
        
        for log_file in log_files:
            entries = list(self.read_log_file(log_file, since))
            all_entries.extend(entries)
        
        if not all_entries:
            return {"message": "No recent log entries found"}
        
        # Analyze patterns
        analysis = self.analyze_patterns(all_entries)
        
        # Check for anomalies
        alerts = self.check_anomalies(analysis)
        
        # Process alerts
        for alert in alerts:
            self.alert_manager.process_alert(alert)
        
        analysis['alerts_generated'] = len(alerts)
        analysis['analysis_period'] = f"Last {minutes} minutes"
        
        return analysis
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        since = datetime.now() - timedelta(hours=hours)
        error_log = Path("logs/errors.log")
        
        if not error_log.exists():
            return {"message": "No error log file found"}
        
        error_entries = list(self.read_log_file(error_log, since))
        
        if not error_entries:
            return {"message": f"No errors found in the last {hours} hours"}
        
        # Group errors by hour
        hourly_errors = defaultdict(int)
        error_types = Counter()
        
        for entry in error_entries:
            try:
                timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                hourly_errors[hour_key] += 1
                
                # Categorize error type
                message = entry.get('message', '').lower()
                if 'timeout' in message:
                    error_types['timeout'] += 1
                elif 'connection' in message:
                    error_types['connection'] += 1
                elif 'memory' in message:
                    error_types['memory'] += 1
                elif 'authentication' in message or 'auth' in message:
                    error_types['authentication'] += 1
                else:
                    error_types['other'] += 1
                    
            except (ValueError, TypeError):
                continue
        
        return {
            "total_errors": len(error_entries),
            "period": f"Last {hours} hours",
            "hourly_breakdown": dict(hourly_errors),
            "error_types": dict(error_types),
            "most_recent_errors": error_entries[-5:]  # Last 5 errors
        }
