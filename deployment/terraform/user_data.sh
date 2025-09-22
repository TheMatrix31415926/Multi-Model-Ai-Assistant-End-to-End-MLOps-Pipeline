# deployment/terraform/user_data.sh - EC2 initialization script
#!/bin/bash
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install Git
yum install -y git

# Install Python and pip
yum install -y python3 python3-pip

# Install AWS CLI
pip3 install awscli

# Create application directory
mkdir -p /home/ec2-user/${project_name}
chown ec2-user:ec2-user /home/ec2-user/${project_name}

# Create systemd service for the application
cat << 'EOF' > /etc/systemd/system/multimodal-ai.service
[Unit]
Description=Multi-Modal AI Assistant
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/multimodal-ai-assistant
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ec2-user
Group=ec2-user

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable multimodal-ai

# Create deployment script
cat << 'EOF' > /home/ec2-user/deploy.sh
#!/bin/bash
cd /home/ec2-user/multimodal-ai-assistant

# Pull latest code (you'll need to set up your repo)
# git pull origin main

# Stop existing containers
docker-compose down

# Build and start containers
docker-compose up -d

# Show status
docker-compose ps
EOF

chmod +x /home/ec2-user/deploy.sh
chown ec2-user:ec2-user /home/ec2-user/deploy.sh