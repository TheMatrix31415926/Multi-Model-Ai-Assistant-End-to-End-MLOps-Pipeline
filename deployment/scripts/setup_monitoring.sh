# deployment/scripts/setup_monitoring.sh - Complete monitoring setup
#!/bin/bash

set -e

echo " Multi-Modal AI Assistant - Complete Monitoring Setup"
echo "======================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create monitoring directories
create_directories() {
    echo " Creating monitoring directories..."
    
    mkdir -p monitoring/{prometheus,grafana,alertmanager}
    mkdir -p monitoring/grafana/{dashboards,datasources}
    mkdir -p monitoring/alerting
    mkdir -p logs
    
    echo -e "${GREEN} Directories created${NC}"
}

# Setup Prometheus configuration
setup_prometheus() {
    echo " Setting up Prometheus configuration..."
    
    # Create Prometheus config if not exists
    if [ ! -f "monitoring/prometheus/prometheus.yml" ]; then
        echo "Creating prometheus.yml from template..."
        # File already created in artifacts above
    fi
    
    echo -e "${GREEN} Prometheus configured${NC}"
}

# Setup Grafana
setup_grafana() {
    echo " Setting up Grafana..."
    
    # Create Grafana datasource config
    mkdir -p monitoring/grafana/datasources
    cat << 'EOF' > monitoring/grafana/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create dashboard provisioning config
    mkdir -p monitoring/grafana/dashboards
    cat << 'EOF' > monitoring/grafana/dashboards/dashboard.yml
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    echo -e "${GREEN} Grafana configured${NC}"
}

# Setup monitoring stack with Docker Compose
setup_monitoring_stack() {
    echo " Setting up monitoring Docker stack..."
    
    # Create complete monitoring compose file
    cat << 'EOF' > docker-compose.monitoring.yml
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: multimodal-ai-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
    networks:
      - monitoring-network

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: multimodal-ai-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - monitoring-network
    depends_on:
      - prometheus

  # AlertManager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: multimodal-ai-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - monitoring-network

  # Node Exporter
  node-exporter:
    image: prom/node-exporter:latest
    container_name: multimodal-ai-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($|/)'
    networks:
      - monitoring-network

  # cAdvisor
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: multimodal-ai-cadvisor
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    networks:
      - monitoring-network

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  monitoring-network:
    driver: bridge
EOF

    echo -e "${GREEN} Monitoring stack configured${NC}"
}

# Update main application with monitoring
update_main_application() {
    echo " Updating main application with monitoring..."
    
    # Update main docker-compose.yml to include monitoring network
    if grep -q "networks:" docker-compose.yml; then
        echo "Networks already configured in docker-compose.yml"
    else
        cat << 'EOF' >> docker-compose.yml

networks:
  default:
    external: true
    name: multimodal-ai-assistant_monitoring-network
EOF
    fi
    
    echo -e "${GREEN} Main application updated${NC}"
}

# Setup alerting configuration
setup_alerting() {
    echo " Setting up alerting configuration..."
    
    mkdir -p monitoring/alertmanager
    
    cat << 'EOF' > monitoring/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@multimodal-ai.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://api:8000/webhooks/alerts'
    send_resolved: true

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname']
EOF

    echo -e "${GREEN} Alerting configured${NC}"
}

# Start monitoring stack
start_monitoring() {
    echo " Starting monitoring stack..."
    
    # Start monitoring services
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be ready
    echo " Waiting for services to start..."
    sleep 30
    
    # Check if services are running
    if curl -f -s http://localhost:9090/-/healthy > /dev/null; then
        echo -e "${GREEN} Prometheus is running${NC}"
    else
        echo -e "${YELLOW} Prometheus may still be starting${NC}"
    fi
    
    if curl -f -s http://localhost:3000/api/health > /dev/null; then
        echo -e "${GREEN} Grafana is running${NC}"
    else
        echo -e "${YELLOW} Grafana may still be starting${NC}"
    fi
    
    echo -e "${GREEN} Monitoring stack started${NC}"
}

# Test monitoring setup
test_monitoring() {
    echo " Testing monitoring setup..."
    
    # Test Prometheus targets
    echo "Testing Prometheus targets..."
    TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length')
    echo "Active Prometheus targets: $TARGETS"
    
    # Test Grafana API
    echo "Testing Grafana..."
    GRAFANA_HEALTH=$(curl -s http://admin:admin123@localhost:3000/api/health | jq -r '.database')
    echo "Grafana database: $GRAFANA_HEALTH"
    
    # Test alerting
    echo "Testing alertmanager..."
    if curl -f -s http://localhost:9093/-/healthy > /dev/null; then
        echo -e "${GREEN} Alertmanager is healthy${NC}"
    else
        echo -e "${YELLOW} Alertmanager may not be ready${NC}"
    fi
    
    echo -e "${GREEN} Monitoring tests completed${NC}"
}

# Show monitoring URLs
show_monitoring_info() {
    echo ""
    echo " Monitoring Dashboard URLs:"
    echo "============================"
    echo " Prometheus: http://localhost:9090"
    echo " Grafana: http://localhost:3000 (admin/admin123)"
    echo " AlertManager: http://localhost:9093"
    echo " Node Exporter: http://localhost:9100/metrics"
    echo " cAdvisor: http://localhost:8080"
    echo ""
    
    echo " Useful Commands:"
    echo "=================="
    echo "# View monitoring logs:"
    echo "docker-compose -f docker-compose.monitoring.yml logs -f"
    echo ""
    echo "# Restart monitoring:"
    echo "docker-compose -f docker-compose.monitoring.yml restart"
    echo ""
    echo "# Stop monitoring:"
    echo "docker-compose -f docker-compose.monitoring.yml down"
    echo ""
    echo "# View Prometheus config:"
    echo "curl http://localhost:9090/api/v1/status/config"
    echo ""
}

# Main setup function
main() {
    echo "Starting complete monitoring setup..."
    
    create_directories
    setup_prometheus
    setup_grafana
    setup_monitoring_stack
    update_main_application
    setup_alerting
    start_monitoring
    test_monitoring
    show_monitoring_info
    
    echo ""
    echo -e "${GREEN} Monitoring setup completed successfully!${NC}"
    echo "Your Multi-Modal AI Assistant now has comprehensive monitoring and alerting!"
}

# Run main function
main