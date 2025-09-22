# deployment/scripts/deploy.sh - Complete deployment script
#!/bin/bash

set -e

echo "🚀 Multi-Modal AI Assistant - Docker Deployment Script"
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
PULL_IMAGES="true"
BUILD_IMAGES="true"
RUN_TESTS="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-pull)
            PULL_IMAGES="false"
            shift
            ;;
        --no-build)
            BUILD_IMAGES="false"
            shift
            ;;
        --no-tests)
            RUN_TESTS="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env [dev|prod]    Set environment (default: dev)"
            echo "  --no-pull          Skip pulling base images"
            echo "  --no-build         Skip building containers"
            echo "  --no-tests         Skip running tests"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Environment: $ENVIRONMENT${NC}"
echo -e "${BLUE}Pull Images: $PULL_IMAGES${NC}"
echo -e "${BLUE}Build Images: $BUILD_IMAGES${NC}"
echo -e "${BLUE}Run Tests: $RUN_TESTS${NC}"
echo ""

# Function to check if docker is running
check_docker() {
    echo "🔍 Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Function to check if docker-compose is available
check_docker_compose() {
    echo "🔍 Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            echo -e "${RED}❌ Docker Compose is not available${NC}"
            exit 1
        else
            DOCKER_COMPOSE_CMD="docker compose"
        fi
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    echo -e "${GREEN}✅ Docker Compose is available${NC}"
}

# Function to pull base images
pull_images() {
    if [ "$PULL_IMAGES" = "true" ]; then
        echo "📥 Pulling base images..."
        docker pull python:3.9-slim
        docker pull mongo:6.0
        docker pull chromadb/chroma:latest
        docker pull nginx:alpine
        echo -e "${GREEN}✅ Base images pulled${NC}"
    else
        echo "⏭️ Skipping image pull"
    fi
}

# Function to build containers
build_containers() {
    if [ "$BUILD_IMAGES" = "true" ]; then
        echo "🔨 Building containers..."
        if [ "$ENVIRONMENT" = "prod" ]; then
            $DOCKER_COMPOSE_CMD build --no-cache
        else
            $DOCKER_COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml build
        fi
        echo -e "${GREEN}✅ Containers built${NC}"
    else
        echo "⏭️ Skipping container build"
    fi
}

# Function to stop existing containers
stop_existing() {
    echo "🛑 Stopping existing containers..."
    $DOCKER_COMPOSE_CMD down --remove-orphans
    if [ "$ENVIRONMENT" = "dev" ]; then
        $DOCKER_COMPOSE_CMD -f docker-compose.dev.yml down --remove-orphans 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Existing containers stopped${NC}"
}

# Function to start services
start_services() {
    echo "🚀 Starting services..."
    
    if [ "$ENVIRONMENT" = "prod" ]; then
        $DOCKER_COMPOSE_CMD --profile production up -d
    else
        $DOCKER_COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml up -d
    fi
    
    echo -e "${GREEN}✅ Services started${NC}"
}

# Function to wait for services
wait_for_services() {
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for API
    echo "Waiting for API..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}✅ API is ready${NC}"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo -e "${RED}❌ API failed to start within timeout${NC}"
        return 1
    fi
    
    # Wait for Frontend
    echo "Waiting for Frontend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f -s http://localhost:8501 > /dev/null; then
            echo -e "${GREEN}✅ Frontend is ready${NC}"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo -e "${YELLOW}⚠️ Frontend may not be fully ready${NC}"
    fi
}

# Function to run tests
run_tests() {
    if [ "$RUN_TESTS" = "true" ]; then
        echo "🧪 Running tests..."
        sleep 5  # Give services more time
        
        if python3 deployment/scripts/test_docker.py; then
            echo -e "${GREEN}✅ All tests passed${NC}"
        else
            echo -e "${YELLOW}⚠️ Some tests failed, but deployment continues${NC}"
        fi
    else
        echo "⏭️ Skipping tests"
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "📊 Container Status:"
    $DOCKER_COMPOSE_CMD ps
    
    echo ""
    echo "🌐 Service URLs:"
    echo "   - API Documentation: http://localhost:8000/docs"
    echo "   - API Health: http://localhost:8000/health"
    echo "   - Frontend App: http://localhost:8501"
    echo "   - ChromaDB: http://localhost:8002"
    echo "   - MLflow UI: http://localhost:5000"
    
    if [ "$ENVIRONMENT" = "prod" ]; then
        echo "   - Nginx Proxy: http://localhost:80"
    fi
    
    echo ""
    echo "📋 Useful Commands:"
    echo "   - View logs: $DOCKER_COMPOSE_CMD logs -f"
    echo "   - Stop services: $DOCKER_COMPOSE_CMD down"
    echo "   - Restart services: $DOCKER_COMPOSE_CMD restart"
}

# Main deployment flow
main() {
    echo "Starting deployment..."
    
    check_docker
    check_docker_compose
    pull_images
    stop_existing
    build_containers
    start_services
    wait_for_services
    run_tests
    show_status
    
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}Your Multi-Modal AI Assistant is now running.${NC}"
}

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    $DOCKER_COMPOSE_CMD logs > deployment_logs.txt 2>&1 || true
    echo "Logs saved to deployment_logs.txt"
}

trap cleanup EXIT

# Run main function
main