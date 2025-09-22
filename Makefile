install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/

lint:
	flake8 .
	black . --check

format:
	black .
	isort .

run:
	python app.py

run-api:
	uvicorn api.main:app --reload

run-frontend:
	streamlit run frontend/app.py

docker-build:
	docker build -t multimodal-ai-assistant .

docker-run:
	docker run -p 8000:8000 multimodal-ai-assistant

clean:
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete


# Makefile - Docker management commands
.PHONY: help build up down logs clean dev prod

help: ## Show this help message
	@echo "Multi-Modal AI Assistant - Docker Commands"
	@echo "=========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all containers
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

logs-frontend: ## View frontend logs
	docker-compose logs -f frontend

dev: ## Start development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

prod: ## Start production environment
	docker-compose --profile production up -d

restart: ## Restart all services
	docker-compose restart

ps: ## Show running containers
	docker-compose ps

clean: ## Clean up containers and volumes
	docker-compose down -v --remove-orphans
	docker system prune -f

health: ## Check health of all services
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@curl -s http://localhost:8501/_stcore/health || echo "Frontend not responding"
	@curl -s http://localhost:8002/api/v1/heartbeat || echo "ChromaDB not responding"

build-no-cache: ## Build without cache
	docker-compose build --no-cache

update: ## Update and rebuild services
	git pull
	docker-compose build
	docker-compose up -d


.PHONY: help aws-deploy aws-test aws-cleanup k8s-deploy monitor costs

help: ## Show help
	@echo "Multi-Modal AI Assistant - Phase 8 (AWS Deployment)"
	@echo "===================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $1, $2}' $(MAKEFILE_LIST)

aws-setup: ## Complete AWS setup (infrastructure + application)
	chmod +x deployment/scripts/*.sh
	./deployment/scripts/complete_aws_setup.sh

aws-quick: ## Quick AWS deployment (minimal setup)
	chmod +x deployment/scripts/*.sh
	./deployment/scripts/deploy_to_aws.sh

aws-test: ## Test AWS deployment
	python3 deployment/scripts/quick_test.py

aws-cleanup: ## Cleanup all AWS resources
	cd deployment/terraform && terraform destroy -auto-approve

k8s-deploy: ## Deploy to Kubernetes
	chmod +x deployment/scripts/deploy_k8s.sh
	./deployment/scripts/deploy_k8s.sh

monitor: ## Monitor AWS costs
	./deployment/scripts/cost_monitor.sh

github-secrets: ## Show GitHub secrets setup
	./deployment/scripts/setup_github_secrets.sh

logs: ## View application logs (requires SSH)
	@echo "SSH to your instance and run: docker-compose logs -f"

status: ## Check AWS resources status
	@echo "Checking Terraform state..."
	@cd deployment/terraform && terraform show 2>/dev/null || echo "No Terraform state found"
    
.PHONY: help monitoring start-monitoring stop-monitoring monitoring-logs dashboard

help: ## Show help
	@echo "Multi-Modal AI Assistant - Phase 9 (Monitoring & Observability)"
	@echo "=================================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $1, $2}' $(MAKEFILE_LIST)

setup-monitoring: ## Setup complete monitoring stack
	chmod +x deployment/scripts/setup_monitoring.sh
	./deployment/scripts/setup_monitoring.sh

start-monitoring: ## Start monitoring services only
	docker-compose -f docker-compose.monitoring.yml up -d

stop-monitoring: ## Stop monitoring services
	docker-compose -f docker-compose.monitoring.yml down

restart-monitoring: ## Restart monitoring services
	docker-compose -f docker-compose.monitoring.yml restart

monitoring-logs: ## View monitoring logs
	docker-compose -f docker-compose.monitoring.yml logs -f

start-all: ## Start application + monitoring
	docker-compose up -d
	docker-compose -f docker-compose.monitoring.yml up -d

stop-all: ## Stop everything
	docker-compose down
	docker-compose -f docker-compose.monitoring.yml down

dashboard: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo " Prometheus: http://localhost:9090"
	@echo " Grafana: http://localhost:3000 (admin/admin123)"
	@echo " AlertManager: http://localhost:9093"
	@echo " Application Dashboard: http://localhost:8000/monitoring/dashboard"

health: ## Check all services health
	@echo " Checking service health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "API not responding"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus not responding"
	@curl -s http://localhost:3000/api/health || echo "Grafana not responding"

test-monitoring: ## Test monitoring setup
	@echo " Testing monitoring components..."
	@python3 -c "
import requests
import json
import time

print('Testing API metrics...')
try:
    response = requests.get('http://localhost:8000/metrics', timeout=5)
    if response.status_code == 200:
        print(' API metrics endpoint working')
    else:
        print(' API metrics failed')
except Exception as e:
    print(f' API metrics error: {e}')

print('Testing Prometheus...')
try:
    response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
    if response.status_code == 200:
        data = response.json()
        targets = len(data['data']['activeTargets'])
        print(f' Prometheus has {targets} active targets')
    else:
        print(' Prometheus targets failed')
except Exception as e:
    print(f' Prometheus error: {e}')

print('Testing Grafana...')
try:
    response = requests.get('http://localhost:3000/api/health', timeout=5)
    if response.status_code == 200:
        print(' Grafana is healthy')
    else:
        print(' Grafana health check failed')
except Exception as e:
    print(f' Grafana error: {e}')

print(' Monitoring test completed!')
"

monitor-resources: ## Monitor system resources
	@echo " System Resource Usage:"
	@echo "========================"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

view-alerts: ## View recent alerts
	@echo " Recent Alerts:"
	@echo "================"
	@curl -s http://localhost:9093/api/v1/alerts | jq '.data[] | {alertname: .labels.alertname, severity: .labels.severity, status: .status.state}' 2>/dev/null || echo "No alerts or AlertManager not running"

cleanup-monitoring: ## Clean up monitoring data
	docker-compose -f docker-compose.monitoring.yml down -v
	docker volume prune -f


# Makefile - Phase 10 testing commands
.PHONY: help test test-unit test-integration test-e2e test-all coverage

help: ## Show help for Phase 10 (Testing)
	@echo "Multi-Modal AI Assistant - Phase 10 (Testing & Quality)"
	@echo "======================================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $1, $2}' $(MAKEFILE_LIST)

install-test-deps: ## Install testing dependencies
	pip install -r requirements-test.txt

test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	python -m pytest tests/integration/ -v --tb=short

test-e2e: ## Run end-to-end tests only
	python -m pytest tests/e2e/ -v --tb=short -m "not slow"

test-e2e-slow: ## Run all E2E tests including slow ones
	python -m pytest tests/e2e/ -v --tb=short

test-all: ## Run all tests
	python -m pytest tests/ -v --tb=short -m "not slow"

test-fast: ## Run fast tests only (exclude slow and E2E)
	python -m pytest tests/unit/ tests/integration/ -v --tb=short

coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=multimodal_ai_assistant --cov=api --cov=monitoring --cov-report=html --cov-report=term

coverage-unit: ## Unit test coverage only
	python -m pytest tests/unit/ --cov=multimodal_ai_assistant --cov-report=html --cov-report=term

lint: ## Run code linting
	flake8 multimodal_ai_assistant/ api/ monitoring/ tests/
	black --check multimodal_ai_assistant/ api/ monitoring/ tests/
	isort --check-only multimodal_ai_assistant/ api/ monitoring/ tests/

format: ## Format code
	black multimodal_ai_assistant/ api/ monitoring/ tests/
	isort multimodal_ai_assistant/ api/ monitoring/ tests/

test-quality: ## Run quality checks
	make lint
	make test-fast
	make coverage-unit

test-ci: ## Run tests suitable for CI/CD
	python -m pytest tests/unit/ tests/integration/ -v --tb=short --cov=multimodal_ai_assistant --cov-report=xml

test-local: ## Run comprehensive local testing
	make test-all
	make coverage
	make lint

test-deployment: ## Test deployment-related functionality
	python -m pytest tests/e2e/test_deployment_scenarios.py -v

test-workflows: ## Test complete user workflows
	python -m pytest tests/e2e/test_complete_workflows.py -v

test-scenarios: ## Test realistic user scenarios  
	python -m pytest tests/e2e/test_user_scenarios.py -v

clean-test: ## Clean test artifacts
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

test-report: ## Generate comprehensive test report
	@echo " Generating comprehensive test report..."
	python -m pytest tests/ --html=test_report.html --self-contained-html
	@echo " Test report generated: test_report.html"