# deployment/scripts/setup_github_secrets.sh - Helper script for GitHub secrets
#!/bin/bash

echo " GitHub Secrets Setup Helper"
echo "============================="

echo "You need to set the following secrets in your GitHub repository:"
echo "Go to: Settings > Secrets and variables > Actions"
echo ""

echo "Required secrets:"
echo "- AWS_ACCESS_KEY_ID: Your AWS access key"
echo "- AWS_SECRET_ACCESS_KEY: Your AWS secret key"
echo "- EC2_PRIVATE_KEY: Your EC2 private key (.pem file content)"
echo "- EC2_HOSTNAME: Your EC2 instance public IP/DNS"
# echo "- SLACK_WEBHOOK: (Optional) Slack webhook for notifications"
echo ""

echo " To get your EC2 private key content:"
echo "cat multimodal-ai-key.pem | base64 -w 0"
echo ""

echo " To get your EC2 hostname:"
echo "cd deployment/terraform && terraform output instance_public_dns"