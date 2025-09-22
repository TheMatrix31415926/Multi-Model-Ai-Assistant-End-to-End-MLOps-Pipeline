# deployment/scripts/create_key_pair.sh - Create EC2 key pair
#!/bin/bash

KEY_NAME="multimodal-ai-key"
REGION="us-east-1"

echo " Creating EC2 Key Pair..."

# Create key pair
aws ec2 create-key-pair \
    --key-name $KEY_NAME \
    --region $REGION \
    --query 'KeyMaterial' \
    --output text > ${KEY_NAME}.pem

# Set proper permissions
chmod 600 ${KEY_NAME}.pem

echo " Key pair created: ${KEY_NAME}.pem"
echo " Keep this file safe - you'll need it to SSH into your EC2 instance!"
