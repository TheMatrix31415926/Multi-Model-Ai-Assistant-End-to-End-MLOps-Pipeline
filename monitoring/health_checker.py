# monitoring/health_checker.py - Health monitoring system
import asyncio
import aiohttp
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
from multimodal_ai_assistant.logger import get_logger
from monitoring.alerting.alert_manager import AlertManager, Alert

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_type: str  # 'http', 'tcp', 'process', 'custom'
    target: str
    timeout: int = 10
    interval: int = 30
    retries: int = 3
    expected_status: int = 200

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, alert_manager: AlertManager):
        self.logger = get_logger()
        self.alert_manager = alert_manager
        self.health_checks = []
        self.health_status = {}
        self.running = False
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks"""
        default_checks = [
            HealthCheck("API Health", "http", "http://localhost:8000/health"),
            HealthCheck("Frontend", "http", "http://localhost:8501"),
            HealthCheck("Prometheus", "http", "http://localhost:9090/-/healthy"),
            HealthCheck("Grafana", "http", "http://localhost:3000/api/health"),
            HealthCheck("MongoDB", "tcp", "localhost:27017"),
            HealthCheck("ChromaDB", "http", "http://localhost:8002/api/v1/heartbeat"),
        ]
        
        self.health_checks.extend(default_checks)
    
    def add_health_check(self, health_check: HealthCheck):
        """Add custom health check"""
        self.health_checks.append(health_check)
        self.logger.info(f"Added health check: {health_check.name}")
    
    async def check_http_endpoint(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Check HTTP endpoint health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=health_check.timeout)) as session:
                async with session.get(health_check.target) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == health_check.expected_status:
                        return {
                            "status": "healthy",
                            "response_time": response_time,
                            "status_code": response.status
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "response_time": response_time,
                            "status_code": response.status,
                            "error": f"Expected status {health_check.expected_status}, got {response.status}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "response_time": health_check.timeout,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def check_tcp_port(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Check TCP port connectivity"""
        start_time = time.time()
        
        try:
            host, port = health_check.target.split(':')
            port = int(port)
            
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=health_check.timeout)
            
            writer.close()
            await writer.wait_closed()
            
            return {
                "status": "healthy",
                "response_time": time.time() - start_time
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "response_time": health_check.timeout,
                "error": "Connection timeout"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def perform_health_check(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Perform individual health check with retries"""
        for attempt in range(health_check.retries):
            try:
                if health_check.check_type == "http":
                    result = await self.check_http_endpoint(health_check)
                elif health_check.check_type == "tcp":
                    result = await self.check_tcp_port(health_check)
                else:
                    result = {"status": "unknown", "error": "Unsupported check type"}
                
                # If healthy, return immediately
                if result["status"] == "healthy":
                    result["attempt"] = attempt + 1
                    return result
                
                # If unhealthy and not last attempt, wait before retry
                if attempt < health_check.retries - 1:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                if attempt == health_check.retries - 1:
                    return {
                        "status": "unhealthy",
                        "error": str(e),
                        "attempt": attempt + 1
                    }
                await asyncio.sleep(2)
        
        return result
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = None
            
            system_status = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "process_count": process_count,
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
            
            if load_avg:
                system_status["load_average"] = load_avg
            
            # Check thresholds and create alerts
            if cpu_percent > 80:
                self.alert_manager.create_system_alert("CPU Usage", cpu_percent, 80)
            
            if memory_percent > 85:
                self.alert_manager.create_system_alert("Memory Usage", memory_percent, 85)
            
            if disk_percent > 85:
                self.alert_manager.create_system_alert("Disk Usage", disk_percent, 85)
            
            return system_status
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {"error": str(e)}
    
    async def run_health_checks(self):
        """Run all health checks"""
        tasks = []
        
        for health_check in self.health_checks:
            task = asyncio.create_task(self.perform_health_check(health_check))
            tasks.append((health_check.name, task))
        
        # Wait for all checks to complete
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                
                # Create alerts for unhealthy services
                if result["status"] == "unhealthy":
                    alert = Alert(
                        name=f"Service Unhealthy: {name}",
                        severity="critical" if name in ["API Health", "Frontend"] else "warning",
                        message=f"Health check failed: {result.get('error', 'Unknown error')}",
                        timestamp=datetime.now(),
                        source="health_monitor",
                        metadata={
                            "service": name,
                            "check_result": result
                        }
                    )
                    self.alert_manager.process_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
        
        return results
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting health monitoring loop")
        self.running = True
        
        while self.running:
            try:
                # Run health checks
                health_results = await self.run_health_checks()
                
                # Check system resources
                system_results = self.check_system_resources()
                
                # Update health status
                self.health_status = {
                    "timestamp": datetime.now().isoformat(),
                    "services": health_results,
                    "system": system_results,
                    "overall_status": "healthy" if all(
                        result.get("status") == "healthy" 
                        for result in health_results.values()
                    ) else "degraded"
                }
                
                # Log summary
                healthy_count = sum(1 for result in health_results.values() if result.get("status") == "healthy")
                total_count = len(health_results)
                
                self.logger.info(
                    f"Health check summary: {healthy_count}/{total_count} services healthy",
                    extra={
                        "healthy_services": healthy_count,
                        "total_services": total_count,
                        "system_cpu": system_results.get("cpu_usage"),
                        "system_memory": system_results.get("memory_usage")
                    }
                )
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def start_monitoring(self):
        """Start monitoring in background"""
        if not self.running:
            asyncio.create_task(self.monitoring_loop())
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.logger.info("Health monitoring stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status
