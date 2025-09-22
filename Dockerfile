# FROM python:3.9-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY . .

# EXPOSE 8000

# CMD ["python", "app.py"]



# Dockerfile - Main application container

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/data_ingestion artifacts/data_validation artifacts/data_transformation
RUN mkdir -p logs models experiments

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "deployment/scripts/start_services.py"]