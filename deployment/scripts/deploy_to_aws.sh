# deployment/scripts/deploy_to_aws.sh - Complete AWS deployment
#!/bin/bash

set -e

echo " Multi-Modal AI Assistant - AWS Deployment"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check dependencies
check_dependencies() {
    echo " Checking dependencies..."
    
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED} Terraform not found. Please install Terraform.${NC}"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        echo -e "${RED} AWS CLI not found. Please install AWS CLI.${NC}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED} AWS credentials not configured. Run 'aws configure'${NC}"
        exit 1
    fi
    
    echo -e "${GREEN} All dependencies found${NC}"
}

# Create key pair if it doesn't exist
create_key_pair() {
    echo " Checking EC2 key pair..."
    
    if ! aws ec2 describe-key-pairs --key-names multimodal-ai-key --region us-east-1 &> /dev/null; then
        echo "Creating EC2 key pair..."
        ./deployment/scripts/create_key_pair.sh
    else
        echo -e "${GREEN} Key pair already exists${NC}"
    fi
}

# Initialize and apply Terraform
deploy_infrastructure() {
    echo " Deploying AWS infrastructure..."
    
    cd deployment/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply deployment
    echo "Applying Terraform configuration..."
    terraform apply tfplan
    
    echo -e "${GREEN} Infrastructure deployed${NC}"
    
    # Get outputs
    INSTANCE_IP=$(terraform output -raw instance_public_ip)
    INSTANCE_DNS=$(terraform output -raw instance_public_dns)
    S3_BUCKET=$(terraform output -raw s3_bucket_name)
    
    echo ""
    echo " Deployment Information:"
    echo "Instance IP: $INSTANCE_IP"
    echo "Instance DNS: $INSTANCE_DNS"
    echo "S3 Bucket: $S3_BUCKET"
    
    cd ../..
}

# Deploy application to EC2
deploy_application() {
    echo " Deploying application to EC2..."
    
    INSTANCE_IP=$(cd deployment/terraform && terraform output -raw instance_public_ip)
    
    # Wait for instance to be ready
    echo " Waiting for EC2 instance to be ready..."
    sleep 60
    
    # Create deployment package
    echo " Creating deployment package..."
    tar -czf multimodal-ai-app.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='node_modules' \
        --exclude='.env' \
        api/ frontend/ docker-compose.yml requirements.txt
    
    # Copy application to EC2
    echo " Copying application to EC2..."
    scp -i multimodal-ai-key.pem -o StrictHostKeyChecking=no \
        multimodal-ai-app.tar.gz ec2-user@$INSTANCE_IP:/home/ec2-user/
    
    # Extract and deploy on EC2
    echo " Setting up application on EC2..."
    ssh -i multimodal-ai-key.pem -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        # Extract application
        cd /home/ec2-user
        tar -xzf multimodal-ai-app.tar.gz
        rm multimodal-ai-app.tar.gz
        
        # Start application
        cd /home/ec2-user
        sudo systemctl start multimodal-ai
        
        # Check status
        sleep 10
        docker-compose ps
EOF
    
    # Clean up
    rm multimodal-ai-app.tar.gz
    
    echo -e "${GREEN} Application deployed${NC}"
}

# Test deployment
test_deployment() {
    echo " Testing deployment..."
    
    INSTANCE_IP=$(cd deployment/terraform && terraform output -raw instance_public_ip)
    
    # Wait for services to start
    echo " Waiting for services to start..."
    sleep 30
    
    # Test API health
    echo "Testing API health..."
    if curl -f -s http://$INSTANCE_IP:8000/health > /dev/null; then
        echo -e "${GREEN} API is healthy${NC}"
    else
        echo -e "${YELLOW} API might still be starting${NC}"
    fi
    
    # Test frontend
    echo "Testing frontend..."
    if curl -f -s http://$INSTANCE_IP:8501 > /dev/null; then
        echo -e "${GREEN} Frontend is accessible${NC}"
    else
        echo -e "${YELLOW} Frontend might still be starting${NC}"
    fi
}

# Show final information
show_results() {
    echo ""
    echo " AWS Deployment Complete!"
    echo "=========================="
    
    INSTANCE_IP=$(cd deployment/terraform && terraform output -raw instance_public_ip)
    INSTANCE_DNS=$(cd deployment/terraform && terraform output -raw instance_public_dns)
    
    echo ""
    echo " Access your application at:"
    echo "   - Frontend: http://$INSTANCE_DNS:8501"
    echo "   - API Docs: http://$INSTANCE_DNS:8000/docs"
    echo "   - API Health: http://$INSTANCE_DNS:8000/health"
    echo "   - MLflow: http://$INSTANCE_DNS:5000"
    
    echo ""
    echo " Management Commands:"
    echo "   - SSH to server: ssh -i multimodal-ai-key.pem ec2-user@$INSTANCE_IP"
    echo "   - View logs: ssh to server then 'docker-compose logs -f'"
    echo "   - Restart app: ssh to server then 'sudo systemctl restart multimodal-ai'"
    
    echo ""
    echo " Cost Information:"
    echo "   - EC2 t2.micro: ~$8.5/month (free tier: 750 hours/month)"
    echo "   - S3 storage: ~$0.023/GB/month (free tier: 5GB)"
    echo "   - Data transfer: ~$0.09/GB (free tier: 1GB/month)"
    
    echo ""
    echo " Remember to:"
    echo "   - Monitor your AWS billing dashboard"
    echo "   - Stop the instance when not needed to save costs"
    echo "   - Run 'terraform destroy' to clean up resources"
}

# Main deployment flow
main() {
    check_dependencies
    create_key_pair
    deploy_infrastructure
    deploy_application
    test_deployment
    show_results
}

# Run deployment
main