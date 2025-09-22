# deployment/scripts/deploy_k8s.sh - Kubernetes deployment script
#!/bin/bash

set -e

echo " Deploying Multi-Modal AI Assistant to Kubernetes"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check dependencies
check_dependencies() {
    echo " Checking dependencies..."
    
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED} kubectl not found. Please install kubectl.${NC}"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED} Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED} Cannot connect to Kubernetes cluster.${NC}"
        echo "Please ensure your kubeconfig is properly set up."
        exit 1
    fi
    
    echo -e "${GREEN} All dependencies found${NC}"
}

# Build and push Docker images
build_and_push_images() {
    echo "🔨 Building Docker images..."
    
    # Build API image
    docker build -f deployment/docker/Dockerfile.api -t multimodal-ai-api:latest .
    
    # Build Frontend image
    docker build -f deployment/docker/Dockerfile.frontend -t multimodal-ai-frontend:latest .
    
    echo -e "${GREEN} Docker images built${NC}"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    echo " Deploying to Kubernetes..."
    
    # Apply manifests in order
    kubectl apply -f deployment/kubernetes/namespace.yaml
    kubectl apply -f deployment/kubernetes/configmap.yaml
    kubectl apply -f deployment/kubernetes/secrets.yaml
    kubectl apply -f deployment/kubernetes/mongodb-deployment.yaml
    kubectl apply -f deployment/kubernetes/chromadb-deployment.yaml
    
    # Wait for databases to be ready
    echo " Waiting for databases to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/mongodb -n multimodal-ai
    kubectl wait --for=condition=available --timeout=300s deployment/chromadb -n multimodal-ai
    
    # Deploy application
    kubectl apply -f deployment/kubernetes/api-deployment.yaml
    kubectl apply -f deployment/kubernetes/frontend-deployment.yaml
    kubectl apply -f deployment/kubernetes/hpa.yaml
    
    # Wait for application to be ready
    echo " Waiting for application to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/multimodal-ai-api -n multimodal-ai
    kubectl wait --for=condition=available --timeout=300s deployment/multimodal-ai-frontend -n multimodal-ai
    
    echo -e "${GREEN} Deployment completed${NC}"
}

# Get service information
get_service_info() {
    echo " Getting service information..."
    
    # Get LoadBalancer IP/DNS
    FRONTEND_LB=$(kubectl get service multimodal-ai-frontend-service -n multimodal-ai -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -z "$FRONTEND_LB" ]; then
        FRONTEND_LB=$(kubectl get service multimodal-ai-frontend-service -n multimodal-ai -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    fi
    
    if [ -z "$FRONTEND_LB" ]; then
        echo -e "${YELLOW} LoadBalancer is still provisioning. Using port-forward for now.${NC}"
        FRONTEND_URL="Use 'kubectl port-forward service/multimodal-ai-frontend-service 8501:8501 -n multimodal-ai'"
        API_URL="Use 'kubectl port-forward service/multimodal-ai-api-service 8000:8000 -n multimodal-ai'"
    else
        FRONTEND_URL="http://$FRONTEND_LB:8501"
        API_URL="http://$FRONTEND_LB:8000"
    fi
    
    echo ""
    echo " Service URLs:"
    echo "   - Frontend: $FRONTEND_URL"
    echo "   - API: $API_URL"
    echo ""
}

# Show cluster status
show_status() {
    echo " Cluster Status:"
    kubectl get pods -n multimodal-ai
    echo ""
    kubectl get services -n multimodal-ai
    echo ""
    kubectl get hpa -n multimodal-ai
}

# Main deployment
main() {
    check_dependencies
    build_and_push_images
    deploy_to_kubernetes
    get_service_info
    show_status
    
    echo ""
    echo -e "${GREEN} Kubernetes deployment completed!${NC}"
    echo ""
    echo " Useful commands:"
    echo "   - View logs: kubectl logs -f deployment/multimodal-ai-api -n multimodal-ai"
    echo "   - Scale API: kubectl scale deployment multimodal-ai-api --replicas=3 -n multimodal-ai"
    echo "   - Port forward: kubectl port-forward service/multimodal-ai-frontend-service 8501:8501 -n multimodal-ai"
    echo "   - Delete deployment: kubectl delete namespace multimodal-ai"
}

main