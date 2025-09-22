# Quick start commands 

# setup_docker.sh - One-command setup
#!/bin/bash

echo "🐳 Setting up Multi-Modal AI Assistant with Docker"

# Make scripts executable
chmod +x deployment/scripts/*.sh
chmod +x deployment/scripts/*.py

# Create necessary directories
mkdir -p logs artifacts/{data_ingestion,data_validation,data_transformation}
mkdir -p chroma_db experiments models

# Run deployment
./deployment/scripts/deploy.sh --env dev

echo "✅ Docker setup complete!"
echo "🌐 Access your app at: http://localhost:8501"