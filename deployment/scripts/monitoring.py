# deployment/scripts/monitoring.py - Container monitoring script
#!/usr/bin/env python3

import docker
import time
import requests
import json
from datetime import datetime
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

class ContainerMonitor:
    def __init__(self):
        self.client = docker.from_env()
        self.running = True
        self.services = {
            'api': {'container': 'multimodal-ai-api', 'port': 8000, 'health_path': '/health'},
            'frontend': {'container': 'multimodal-ai-frontend', 'port': 8501, 'health_path': '/_stcore/health'},
            'mongodb': {'container': 'multimodal-ai-mongodb', 'port': 27017, 'health_path': None},
            'chromadb': {'container': 'multimodal-ai-chromadb', 'port': 8002, 'health_path': '/api/v1/heartbeat'},
            'mlflow': {'container': 'multimodal-ai-mlflow', 'port': 5000, 'health_path': '/health'}
        }
    
    def check_container_status(self, container_name):
        """Check if container is running"""
        try:
            container = self.client.containers.get(container_name)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        except Exception as e:
            logging.error(f"Error checking container {container_name}: {e}")
            return False
    
    def check_service_health(self, service_name, config):
        """Check service health via HTTP"""
        if not config.get('health_path'):
            return True  # Skip health check for services without HTTP endpoint
        
        try:
            url = f"http://localhost:{config['port']}{config['health_path']}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_container_stats(self, container_name):
        """Get container resource usage stats"""
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            else:
                cpu_percent = 0.0
            
            # Calculate memory usage
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats']['limit']
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(mem_usage / 1024 / 1024, 2),
                'memory_percent': round(mem_percent, 2),
                'network_rx_bytes': stats['networks']['eth0']['rx_bytes'] if 'networks' in stats else 0,
                'network_tx_bytes': stats['networks']['eth0']['tx_bytes'] if 'networks' in stats else 0
            }
            
        except Exception as e:
            logging.error(f"Error getting stats for {container_name}: {e}")
            return None
    
    def monitor_services(self):
        """Monitor all services continuously"""
        logging.info("Starting service monitoring...")
        
        while self.running:
            monitor_data = {
                'timestamp': datetime.now().isoformat(),
                'services': {}
            }
            
            for service_name, config in self.services.items():
                container_name = config['container']
                
                # Check container status
                container_running = self.check_container_status(container_name)
                
                # Check service health
                service_healthy = False
                if container_running:
                    service_healthy = self.check_service_health(service_name, config)
                
                # Get resource stats
                stats = None
                if container_running:
                    stats = self.get_container_stats(container_name)
                
                monitor_data['services'][service_name] = {
                    'container_running': container_running,
                    'service_healthy': service_healthy,
                    'stats': stats
                }
                
                # Log status
                status_emoji = "G" if container_running and service_healthy else "R"
                logging.info(f"{status_emoji} {service_name}: Running={container_running}, Healthy={service_healthy}")
            
            # Save monitoring data
            try:
                with open('logs/monitoring_data.json', 'a') as f:
                    f.write(json.dumps(monitor_data) + '\n')
            except Exception as e:
                logging.error(f"Error saving monitoring data: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def generate_health_report(self):
        """Generate health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'services': {},
            'alerts': []
        }
        
        for service_name, config in self.services.items():
            container_name = config['container']
            
            container_running = self.check_container_status(container_name)
            service_healthy = self.check_service_health(service_name, config) if container_running else False
            stats = self.get_container_stats(container_name) if container_running else None
            
            service_status = {
                'status': 'healthy' if container_running and service_healthy else 'unhealthy',
                'container_running': container_running,
                'service_healthy': service_healthy,
                'stats': stats
            }
            
            report['services'][service_name] = service_status
            
            # Check for alerts
            if not container_running:
                report['alerts'].append(f"{service_name} container is not running")
                report['overall_status'] = 'degraded'
            elif not service_healthy:
                report['alerts'].append(f"{service_name} service is not healthy")
                report['overall_status'] = 'degraded'
            elif stats and stats['memory_percent'] > 90:
                report['alerts'].append(f"{service_name} high memory usage: {stats['memory_percent']:.1f}%")
            elif stats and stats['cpu_percent'] > 80:
                report['alerts'].append(f"{service_name} high CPU usage: {stats['cpu_percent']:.1f}%")
        
        return report
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

def main():
    """Main monitoring function"""
    print(" Multi-Modal AI Assistant - Container Monitor")
    print("=" * 50)
    
    monitor = ContainerMonitor()
    
    try:
        # Start continuous monitoring
        monitor_thread = monitor.start_monitoring()
        
        # Keep main thread alive and provide periodic reports
        while True:
            time.sleep(300)  # Generate report every 5 minutes
            
            report = monitor.generate_health_report()
            
            print(f"\n Health Report - {report['timestamp']}")
            print(f"Overall Status: {report['overall_status']}")
            
            if report['alerts']:
                print(" Alerts:")
                for alert in report['alerts']:
                    print(f"  - {alert}")
            else:
                print(" No alerts")
            
    except KeyboardInterrupt:
        print("\n Stopping monitor...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
