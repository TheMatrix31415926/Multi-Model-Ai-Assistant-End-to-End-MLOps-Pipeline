# deployment/scripts/cost_monitor.sh - AWS cost monitoring
#!/bin/bash

echo " AWS Cost Monitor - Multi-Modal AI Assistant"
echo "=============================================="

# Get current AWS account
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)

if [ -z "$ACCOUNT_ID" ]; then
    echo " AWS credentials not configured or invalid"
    exit 1
fi

echo "Account ID: $ACCOUNT_ID"
echo "Region: us-east-1"
echo ""

# Get current month costs
echo " Current Month Costs:"
echo "======================="

# Get billing data (requires billing permissions)
aws ce get-cost-and-usage \
    --time-period Start=$(date -d "$(date +%Y-%m-01)" +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --query 'ResultsByTime[0].Groups[?Metrics.UnblendedCost.Amount>`0.01`].[Keys[0],Metrics.UnblendedCost.Amount]' \
    --output table 2>/dev/null || echo "⚠️ Billing data not available (requires billing permissions)"

echo ""
echo " Free Tier Usage (Estimated):"
echo "==============================="
echo " EC2 t2.micro hours: Check AWS Console > EC2 > Running Instances"
echo " EBS storage: 30GB used of 30GB free tier"
echo " S3 storage: Check S3 Console for usage"
echo " Data transfer: Monitor CloudWatch metrics"

echo ""
echo " Cost Optimization Tips:"
echo "=========================="
echo "1. Stop EC2 instance when not needed: aws ec2 stop-instances --instance-ids [your-instance-id]"
echo "2. Use CloudWatch to monitor usage"
echo "3. Set up billing alerts in AWS Console"
echo "4. Clean up unused S3 objects"
echo "5. Consider using AWS Budgets for cost control"

echo ""
echo " Monitoring Commands:"
echo "======================="
echo "# Check running instances:"
echo "aws ec2 describe-instances --query 'Reservations[].Instances[?State.Name==\`running\`].[InstanceId,InstanceType,LaunchTime]' --output table"
echo ""
echo "# Check S3 bucket size:"
echo "aws s3 ls s3://your-bucket-name --recursive --summarize"
echo ""
echo "# Set up billing alert:"
echo "aws cloudwatch put-metric-alarm --alarm-name 'BillingAlert' --alarm-description 'Alert when AWS bill exceeds \$10' --metric-name EstimatedCharges --namespace AWS/Billing --statistic Maximum --period 86400 --threshold 10 --comparison-operator GreaterThanThreshold"
