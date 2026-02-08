import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project name
project_name = "multimodal_ai_assistant"

# Complete directory structure for Multi-Modal AI Assistant
directories = [
    # Core project structure
    f"{project_name}",
    f"{project_name}/components",
    f"{project_name}/configuration",
    f"{project_name}/constants",
    f"{project_name}/entity",
    f"{project_name}/exception",
    f"{project_name}/logger",
    f"{project_name}/pipeline",
    f"{project_name}/utils",
    
    # Multi-Modal AI Components
    f"{project_name}/multimodal",
    f"{project_name}/multimodal/vision",
    f"{project_name}/multimodal/nlp", 
    f"{project_name}/multimodal/fusion",
    f"{project_name}/multimodal/agents",
    f"{project_name}/multimodal/embeddings",
    
    # RAG & Vector Store
    f"{project_name}/vector_store",
    f"{project_name}/knowledge_base",
    f"{project_name}/retrievers",
    
    # API & Frontend
    "api",
    "api/routers",
    "api/middleware",
    "api/schemas",
    "frontend",
    "frontend/components",
    "frontend/pages",
    "frontend/utils",
    
    # MLOps Pipelines
    "pipelines",
    "pipelines/training",
    "pipelines/inference", 
    "pipelines/data",
    "pipelines/monitoring",
    
    # Models & Experiments
    "models",
    "models/checkpoints",
    "models/artifacts",
    "experiments",
    "experiments/mlflow",
    
    # Configuration
    "configs",
    "configs/model",
    "configs/pipeline",
    "configs/deployment",
    
    # Data directories
    "data",
    "data/raw",
    "data/processed",
    "data/external",
    "data/interim",
    
    # Cloud & Storage
    f"{project_name}/cloud_storage",
    f"{project_name}/data_access",
    
    # Deployment
    "deployment",
    "deployment/docker",
    "deployment/kubernetes",
    "deployment/terraform",
    "deployment/monitoring",
    "deployment/nginx",
    
    # Testing
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/e2e",
    "tests/fixtures",
    
    # CI/CD
    ".github",
    ".github/workflows",
    ".github/templates",
    
    # Scripts & Tools
    "scripts",
    "scripts/setup",
    "scripts/deployment",
    "scripts/data",
    "scripts/monitoring",
    
    # Monitoring & Logging
    "monitoring",
    "monitoring/prometheus",
    "monitoring/grafana",
    "logs",
    
    # Documentation
    "docs",
    "docs/api",
    "docs/deployment",
    "docs/architecture",
    
    # Notebooks
    "notebooks",
    "notebooks/exploration",
    "notebooks/experiments",
    "notebooks/visualization",
    
    # Reports
    "reports",
    "reports/figures",
]

# Files to create with basic structure
files_to_create = [
    # Core project files
    f"{project_name}/__init__.py",
    
    # Component files
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    
    # Configuration files
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/configuration/vector_db_connection.py",
    
    # Multi-modal components
    f"{project_name}/multimodal/__init__.py",
    f"{project_name}/multimodal/vision/__init__.py",
    f"{project_name}/multimodal/vision/image_processor.py",
    f"{project_name}/multimodal/vision/vision_encoder.py",
    f"{project_name}/multimodal/nlp/__init__.py",
    f"{project_name}/multimodal/nlp/text_processor.py",
    f"{project_name}/multimodal/nlp/language_model.py",
    f"{project_name}/multimodal/fusion/__init__.py",
    f"{project_name}/multimodal/fusion/multimodal_fusion.py",
    f"{project_name}/multimodal/agents/__init__.py",
    f"{project_name}/multimodal/agents/vqa_agent.py",
    f"{project_name}/multimodal/agents/conversation_agent.py",
    
    # Vector store & RAG
    f"{project_name}/vector_store/__init__.py",
    f"{project_name}/vector_store/chroma_store.py",
    f"{project_name}/vector_store/embedding_manager.py",
    f"{project_name}/retrievers/__init__.py",
    f"{project_name}/retrievers/document_retriever.py",
    f"{project_name}/retrievers/image_retriever.py",
    
    # Cloud storage
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",
    f"{project_name}/cloud_storage/gcp_storage.py",
    
    # Data access
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/multimodal_data.py",
    
    # Core utilities
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/estimator.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    
    # Pipeline files
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    "pipelines/__init__.py",
    "pipelines/training/__init__.py",
    "pipelines/training/train_multimodal.py",
    "pipelines/inference/__init__.py",
    "pipelines/inference/predict_multimodal.py",
    "pipelines/data/__init__.py",
    "pipelines/data/data_preprocessing.py",
    
    # API files
    "api/__init__.py",
    "api/main.py",
    "api/routers/__init__.py",
    "api/routers/chat.py",
    "api/routers/upload.py",
    "api/routers/health.py",
    "api/schemas/__init__.py",
    "api/schemas/request.py",
    "api/schemas/response.py",
    "api/middleware/__init__.py",
    "api/middleware/auth.py",
    "api/middleware/logging.py",
    
    # Frontend files
    "frontend/__init__.py",
    "frontend/app.py",
    "frontend/components/__init__.py",
    "frontend/components/chat_interface.py",
    "frontend/components/file_upload.py",
    "frontend/pages/__init__.py",
    "frontend/pages/home.py",
    "frontend/pages/chat.py",
    "frontend/utils/__init__.py",
    "frontend/utils/helpers.py",
    
    # Configuration files
    "configs/model.yaml",
    "configs/pipeline.yaml",
    "configs/deployment.yaml",
    "configs/model/vision_config.yaml",
    "configs/model/nlp_config.yaml",
    "configs/model/fusion_config.yaml",
    "configs/pipeline/training_config.yaml",
    "configs/pipeline/inference_config.yaml",
    "configs/deployment/docker_config.yaml",
    "configs/deployment/k8s_config.yaml",
    
    # Test files
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/unit/__init__.py",
    "tests/unit/test_components.py",
    "tests/unit/test_multimodal.py",
    "tests/integration/__init__.py",
    "tests/integration/test_pipeline.py",
    "tests/integration/test_api.py",
    "tests/e2e/__init__.py",
    "tests/e2e/test_full_workflow.py",
    
    # Deployment files
    "deployment/docker/Dockerfile",
    "deployment/docker/docker-compose.yml",
    "deployment/docker/.dockerignore",
    "deployment/kubernetes/deployment.yaml",
    "deployment/kubernetes/service.yaml",
    "deployment/kubernetes/ingress.yaml",
    "deployment/terraform/main.tf",
    "deployment/terraform/variables.tf",
    "deployment/terraform/outputs.tf",
    "deployment/monitoring/prometheus.yml",
    "deployment/monitoring/grafana-dashboard.json",
    
    # CI/CD files
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",
    ".github/workflows/docker-build.yml",
    ".github/workflows/test.yml",
    
    # Scripts
    "scripts/setup/install_dependencies.sh",
    "scripts/setup/setup_environment.sh",
    "scripts/deployment/deploy.sh",
    "scripts/deployment/rollback.sh",
    "scripts/data/download_data.py",
    "scripts/data/preprocess_data.py",
    "scripts/monitoring/health_check.py",
    
    # Root files
    "app.py",
    "main.py",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "pyproject.toml",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "README.md",
    "LICENSE",
    "Makefile",
    ".env.example",
    "demo.py",
    
    # Documentation
    "docs/README.md",
    "docs/api/api_documentation.md",
    "docs/deployment/deployment_guide.md",
    "docs/architecture/system_architecture.md",
    
    # Monitoring
    "monitoring/prometheus/prometheus.yml",
    "monitoring/grafana/dashboard.json",
    
    # Placeholder files
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "data/interim/.gitkeep",
    "models/checkpoints/.gitkeep",
    "models/artifacts/.gitkeep",
    "experiments/mlflow/.gitkeep",
    "logs/.gitkeep",
    "reports/figures/.gitkeep",
]

def create_directories():
    """Create all necessary directories"""
    logging.info("Creating project directories...")
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logging.error(f"❌ Error creating directory {directory}: {str(e)}")

def create_files():
    """Create all necessary files with basic content"""
    logging.info("Creating project files...")
    
    for file_path in files_to_create:
        try:
            # Create parent directories if they don't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist
            if not Path(file_path).exists():
                with open(file_path, 'w') as f:
                    # Add basic content based on file type
                    if file_path.endswith('__init__.py'):
                        f.write('"""Package initialization file"""\n')
                    elif file_path.endswith('.py'):
                        f.write(f'"""{Path(file_path).name} module"""\n\n# TODO: Implement functionality\n')
                    elif file_path.endswith('.md'):
                        f.write(f'# {Path(file_path).stem.replace("_", " ").title()}\n\nTODO: Add documentation\n')
                    elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
                        f.write('# Configuration file\n# TODO: Add configuration\n')
                    elif file_path.endswith('.txt'):
                        f.write('# TODO: Add requirements\n')
                    elif file_path.endswith('.gitkeep'):
                        f.write('# Keep this directory in git\n')
                    elif file_path.endswith('.env.example'):
                        f.write('# Environment variables example\n# Copy to .env and fill in values\n\n# API Keys\nOPENAI_API_KEY=your_openai_key_here\nHUGGINGFACE_TOKEN=your_hf_token_here\n\n# Database\nMONGO_URL=mongodb://localhost:27017\n\n# AWS\nAWS_ACCESS_KEY_ID=your_access_key\nAWS_SECRET_ACCESS_KEY=your_secret_key\nAWS_REGION=us-east-1\n')
                    elif file_path == 'README.md':
                        f.write(f'# {project_name.replace("_", " ").title()}\n\nMulti-Modal AI Assistant with MLOps Pipeline\n\n## Quick Start\n\n```bash\n# Install dependencies\npip install -r requirements.txt\n\n# Run the application\npython app.py\n```\n\n## Features\n\n- Multi-modal AI capabilities (Text + Vision)\n- RAG-powered question answering\n- Complete MLOps pipeline\n- Production-ready deployment\n\n## Documentation\n\nSee `docs/` directory for detailed documentation.\n')
                    elif file_path == 'Dockerfile':
                        f.write('FROM python:3.9-slim\n\nWORKDIR /app\n\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n\nCOPY . .\n\nEXPOSE 8000\n\nCMD ["python", "app.py"]\n')
                    elif file_path == '.gitignore':
                        f.write('# Python\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\nPIPFILE.lock\n\n# Virtual Environment\nvenv/\nenv/\nENV/\n\n# IDE\n.vscode/\n.idea/\n*.swp\n*.swo\n\n# OS\n.DS_Store\nThumbs.db\n\n# Project specific\n.env\nlogs/\nmodels/checkpoints/*\n!models/checkpoints/.gitkeep\ndata/raw/*\n!data/raw/.gitkeep\ndata/processed/*\n!data/processed/.gitkeep\nexperiments/mlflow/*\n!experiments/mlflow/.gitkeep\nartifacts/\n\n# MLflow\nmlruns/\n')
                    elif file_path == 'Makefile':
                        f.write('install:\n\tpip install -r requirements.txt\n\ninstall-dev:\n\tpip install -r requirements-dev.txt\n\ntest:\n\tpytest tests/\n\nlint:\n\tflake8 .\n\tblack . --check\n\nformat:\n\tblack .\n\tisort .\n\nrun:\n\tpython app.py\n\nrun-api:\n\tuvicorn api.main:app --reload\n\nrun-frontend:\n\tstreamlit run frontend/app.py\n\ndocker-build:\n\tdocker build -t multimodal-ai-assistant .\n\ndocker-run:\n\tdocker run -p 8000:8000 multimodal-ai-assistant\n\nclean:\n\tfind . -type d -name __pycache__ -delete\n\tfind . -name "*.pyc" -delete\n')
                    else:
                        f.write('# TODO: Add content\n')
                
                logging.info(f"✅ Created file: {file_path}")
            else:
                logging.info(f"⏭️  File already exists: {file_path}")
                
        except Exception as e:
            logging.error(f"❌ Error creating file {file_path}: {str(e)}")

def main():
    """Main function to create project structure"""
    print(f"🚀 Creating Multi-Modal AI Assistant Project Structure...")
    print(f"📁 Project name: {project_name}")
    print(f"📊 Total directories to create: {len(directories)}")
    print(f"📄 Total files to create: {len(files_to_create)}")
    print("-" * 60)
    
    # Create directories
    create_directories()
    print("-" * 60)
    
    # Create files
    create_files()
    print("-" * 60)
     
    print("✅ Project structure created successfully!")
    print(f"📁 Total directories created: {len(directories)}")
    print(f"📄 Total files created: {len(files_to_create)}")
    print("\n🎯 Next steps:")
    print("1. cd into your project directory")
    print("2. Create virtual environment: python -m venv venv")
    print("3. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Start coding! 🚀")

if __name__ == "__main__":
    main()
