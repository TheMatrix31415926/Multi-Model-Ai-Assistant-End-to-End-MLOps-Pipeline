# deployment/scripts/complete_aws_setup.sh - One-command AWS deployment
#!/bin/bash

set -e

echo " Multi-Modal AI Assistant - Complete AWS Setup"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="multimodal-ai-assistant"
AWS_REGION="us-east-1"
INSTANCE_TYPE="t2.micro"  # Free tier

# Pre-flight checks
check_prerequisites() {
    echo " Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED} AWS CLI not found. Install it first.${NC}"
        echo "Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED} Terraform not found. Install it first.${NC}"
        echo "Install: https://developer.hashicorp.com/terraform/downloads"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED} AWS credentials not configured.${NC}"
        echo "Run: aws configure"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED} Docker not found. Install it first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN} All prerequisites met${NC}"
}

# Estimate costs
show_cost_estimate() {
    echo ""
    echo " AWS Cost Estimate (us-east-1):"
    echo "=================================="
    echo "EC2 t2.micro: FREE (750 hours/month in free tier)"
    echo " EBS 30GB: FREE (30GB in free tier)"
    echo " S3 Storage: FREE (5GB in free tier)"
    echo " Data Transfer: FREE (1GB out/month in free tier)"
    echo " VPC: FREE"
    echo ""
    echo " Total monthly cost in free tier: $0"
    echo " After free tier: ~$8.5/month for t2.micro"
    echo ""
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
}

# Create ECR repositories (for CI/CD)
create_ecr_repositories() {
    echo " Creating ECR repositories..."
    
    # Create API repository
    aws ecr create-repository \
        --repository-name multimodal-ai-api \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        2>/dev/null || echo "API repository already exists"
    
    # Create Frontend repository
    aws ecr create-repository \
        --repository-name multimodal-ai-frontend \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        2>/dev/null || echo "Frontend repository already exists"
    
    echo -e "${GREEN} ECR repositories ready${NC}"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    echo " Deploying AWS infrastructure..."
    
    cd deployment/terraform
    
    # Initialize Terraform
    terraform init
    
    # Create workspace for environment isolation
    terraform workspace new dev 2>/dev/null || terraform workspace select dev
    
    # Plan deployment
    terraform plan \
        -var="project_name=$PROJECT_NAME" \
        -var="aws_region=$AWS_REGION" \
        -var="environment=dev" \
        -var="instance_type=$INSTANCE_TYPE" \
        -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    # Get outputs
    INSTANCE_IP=$(terraform output -raw instance_public_ip)
    INSTANCE_DNS=$(terraform output -raw instance_public_dns)
    S3_BUCKET=$(terraform output -raw s3_bucket_name)
    
    echo ""
    echo " Infrastructure deployed:"
    echo "   Instance IP: $INSTANCE_IP"
    echo "   Instance DNS: $INSTANCE_DNS"
    echo "   S3 Bucket: $S3_BUCKET"
    
    # Save outputs for later use
    cat << EOF > ../aws_outputs.env
export INSTANCE_IP="$INSTANCE_IP"
export INSTANCE_DNS="$INSTANCE_DNS"
export S3_BUCKET="$S3_BUCKET"
EOF
    
    cd ../..
}

# Build and deploy application
deploy_application() {
    echo " Building and deploying application..."
    
    source deployment/aws_outputs.env
    
    # Wait for instance to be fully ready
    echo " Waiting for EC2 instance to be ready (this may take a few minutes)..."
    sleep 120
    
    # Test SSH connectivity
    for i in {1..10}; do
        if ssh -i multimodal-ai-key.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "echo 'Connection test'" 2>/dev/null; then
            echo -e "${GREEN} SSH connection successful${NC}"
            break
        else
            echo "Attempt $i/10: Waiting for SSH..."
            sleep 30
        fi
        
        if [ $i -eq 10 ]; then
            echo -e "${RED} Could not establish SSH connection${NC}"
            exit 1
        fi
    done
    
    # Create deployment package
    echo " Creating deployment package..."
    tar -czf multimodal-ai-deploy.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='node_modules' \
        --exclude='.env' \
        --exclude='deployment/terraform/.terraform' \
        --exclude='deployment/terraform/*.tfstate*' \
        --exclude='*.pem' \
        api/ frontend/ docker-compose.yml requirements.txt deployment/docker/
    
    # Copy to EC2
    echo " Copying application to EC2..."
    scp -i multimodal-ai-key.pem -o StrictHostKeyChecking=no \
        multimodal-ai-deploy.tar.gz ec2-user@$INSTANCE_IP:/home/ec2-user/
    
    # Deploy on EC2
    echo " Setting up application on EC2..."
    ssh -i multimodal-ai-key.pem -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        # Extract application
        tar -xzf multimodal-ai-deploy.tar.gz
        rm multimodal-ai-deploy.tar.gz
        
        # Create environment file
        cat << 'ENVEOF' > .env
# API Configuration
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
MONGO_URL=mongodb://admin:password123@mongodb:27017/multimodal_ai?authSource=admin

# Application Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
PYTHONPATH=/app

# Database Settings
CHROMA_PERSIST_DIR=/home/ec2-user/chroma_db
ENVEOF
        
        # Start services
        docker-compose up -d
        
        # Wait for services to start
        echo " Waiting for services to start..."
        sleep 30
        
        # Show status
        docker-compose ps
        
        echo " Application deployed on EC2!"
EOF
    
    # Cleanup
    rm multimodal-ai-deploy.tar.gz
    
    echo -e "${GREEN} Application deployed successfully${NC}"
}

# Test deployment
test_deployment() {
    echo " Testing deployment..."
    
    source deployment/aws_outputs.env
    
    # Test API health
    echo "Testing API health..."
    for i in {1..5}; do
        if curl -f -s http://$INSTANCE_IP:8000/health > /dev/null; then
            echo -e "${GREEN} API is healthy${NC}"
            API_HEALTHY=true
            break
        else
            echo "Attempt $i/5: API not ready yet..."
            sleep 15
        fi
    done
    
    # Test frontend
    echo "Testing frontend..."
    for i in {1..5}; do
        if curl -f -s http://$INSTANCE_IP:8501 > /dev/null; then
            echo -e "${GREEN} Frontend is accessible${NC}"
            FRONTEND_HEALTHY=true
            break
        else
            echo "Attempt $i/5: Frontend not ready yet..."
            sleep 15
        fi
    done
    
    # Show results
    if [[ "$API_HEALTHY" == "true" && "$FRONTEND_HEALTHY" == "true" ]]; then
        echo -e "${GREEN} All tests passed!${NC}"
    else
        echo -e "${YELLOW} Some services may still be starting up${NC}"
        echo "Check logs with: ssh -i multimodal-ai-key.pem ec2-user@$INSTANCE_IP 'docker-compose logs'"
    fi
}

# Setup monitoring
setup_monitoring() {
    echo " Setting up basic monitoring..."
    
    source deployment/aws_outputs.env
    
    # Create monitoring script on EC2
    ssh -i multimodal-ai-key.pem -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        # Create simple monitoring script
        cat << 'MONEOF' > monitor.sh
#!/bin/bash
echo "$(date): Checking services..."

# Check API
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo " API is healthy"
else
    echo " API is down"
    docker-compose restart api
fi

# Check Frontend
if curl -f -s http://localhost:8501 > /dev/null; then
    echo " Frontend is healthy"
else
    echo " Frontend is down"
    docker-compose restart frontend
fi

# Show resource usage
echo " System resources:"
df -h /
free -h
echo " Docker containers:"
docker-compose ps
MONEOF
        
        chmod +x monitor.sh
        
        # Add to crontab (run every 5 minutes)
        (crontab -l 2>/dev/null || true; echo "*/5 * * * * /home/ec2-user/monitor.sh >> /home/ec2-user/monitor.log 2>&1") | crontab -
        
        echo " Monitoring setup complete"
EOF
}

# Show final results
show_results() {
    echo ""
    echo " AWS Deployment Complete!"
    echo "=========================="
    
    source deployment/aws_outputs.env
    
    echo ""
    echo " Your Multi-Modal AI Assistant is now live at:"
    echo "     Frontend App: http://$INSTANCE_DNS:8501"
    echo "    API Docs: http://$INSTANCE_DNS:8000/docs"
    echo "     API Health: http://$INSTANCE_DNS:8000/health"
    echo ""
    
    echo " Management Commands:"
    echo "   SSH to server: ssh -i multimodal-ai-key.pem ec2-user@$INSTANCE_IP"
    echo "   View app logs: ssh to server, then 'docker-compose logs -f'"
    echo "   Restart app: ssh to server, then 'docker-compose restart'"
    echo "   Stop app: ssh to server, then 'docker-compose down'"
    echo ""
    
    echo " Cost Management:"
    echo "   Current cost: $ (if within free tier limits)"
    echo "   Monitor billing: https://console.aws.amazon.com/billing/"
    echo "   Stop instance: aws ec2 stop-instances --instance-ids [instance-id]"
    echo "   Destroy everything: cd deployment/terraform && terraform destroy"
    echo ""
    
    echo " Security Notes:"
    echo "   - Your API is publicly accessible (demo purposes)"
    echo "   - Add API keys to .env file on the server for full functionality"
    echo "   - Consider adding authentication for production use"
    echo ""
    
    echo " Next Steps:"
    echo "   1. Visit your frontend URL to test the application"
    echo "   2. Add your OpenAI API key for full AI functionality"
    echo "   3. Set up GitHub Actions for automated deployments"
    echo "   4. Configure domain name and SSL certificate"
    echo ""
}

# Setup GitHub Actions secrets helper
setup_github_actions() {
    echo " GitHub Actions Setup Helper"
    echo "============================"
    
    source deployment/aws_outputs.env
    
    echo "To enable automated deployments, add these secrets to your GitHub repository:"
    echo "(Go to: Settings > Secrets and variables > Actions)"
    echo ""
    
    echo "AWS_ACCESS_KEY_ID: [Your AWS Access Key]"
    echo "AWS_SECRET_ACCESS_KEY: [Your AWS Secret Key]"
    echo "EC2_HOSTNAME: $INSTANCE_IP"
    
    echo ""
    echo "EC2_PRIVATE_KEY: (content of multimodal-ai-key.pem file)"
    echo "To get the private key content:"
    echo "cat multimodal-ai-key.pem"
    echo ""
    
    echo "After setting up these secrets, your GitHub Actions will automatically:"
    echo "- Build Docker images on every push"
    echo "- Deploy to your EC2 instance on main branch pushes"
    echo "- Run tests and security scans"
}

# Cleanup function
cleanup_on_failure() {
    echo ""
    echo " Something went wrong. To clean up resources:"
    echo "cd deployment/terraform && terraform destroy"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting complete AWS deployment..."
    echo ""
    
    # Set up cleanup on failure
    trap cleanup_on_failure ERR
    
    check_prerequisites
    show_cost_estimate
    
    # Create key pair if needed
    if [ ! -f "multimodal-ai-key.pem" ]; then
        echo " Creating EC2 key pair..."
        ./deployment/scripts/create_key_pair.sh
    fi
    
    create_ecr_repositories
    deploy_infrastructure
    deploy_application
    test_deployment
    setup_monitoring
    show_results
    setup_github_actions
    
    echo -e "${GREEN} Deployment completed successfully!${NC}"
    echo "Your Multi-Modal AI Assistant is now running on AWS!"
}

# Check if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi