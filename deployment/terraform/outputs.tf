# TODO: Add content
# deployment/terraform/outputs.tf
output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.app_server.public_dns
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.artifacts.bucket
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.app_sg.id
}

output "application_urls" {
  description = "Application URLs"
  value = {
    api_docs    = "http://${aws_instance.app_server.public_dns}:8000/docs"
    frontend    = "http://${aws_instance.app_server.public_dns}:8501"
    api_health  = "http://${aws_instance.app_server.public_dns}:8000/health"
    mlflow      = "http://${aws_instance.app_server.public_dns}:5000"
  }
}