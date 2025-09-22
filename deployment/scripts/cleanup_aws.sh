# deployment/scripts/cleanup_aws.sh - Cleanup script
#!/bin/bash

echo "🧹 Cleaning up AWS resources..."

cd deployment/terraform

# Destroy infrastructure
terraform destroy -auto-approve

echo "✅ AWS resources cleaned up"
echo "💰 This will stop all AWS charges for this project"