
# deployment/docker/entrypoint.sh - Alternative entrypoint script
#!/bin/bash
set -e

echo " Multi-Modal AI Assistant - Docker Entrypoint"
echo "=============================================="

# Wait for any dependencies (if needed)
echo " Checking environment..."

# Check if artifacts directory exists, create if not
if [ ! -d "/app/artifacts" ]; then
    echo " Creating artifacts directory..."
    mkdir -p /app/artifacts/{data_ingestion,data_validation,data_transformation}
fi

# Check if required files exist
if [ ! -f "/app/api/main.py" ]; then
    echo " API main.py not found!"
    exit 1
fi

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Determine what to run based on arguments
case "$1" in
    "api")
        echo " Starting API server only..."
        cd /app
        exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;
    "frontend")
        echo " Starting frontend only..."
        cd /app
        exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    "full"|"")
        echo " Starting full application..."
        exec python /app/deployment/scripts/start_services.py
        ;;
    *)
        echo " Unknown command: $1"
        echo "Usage: $0 [api|frontend|full]"
        exit 1
        ;;
esac