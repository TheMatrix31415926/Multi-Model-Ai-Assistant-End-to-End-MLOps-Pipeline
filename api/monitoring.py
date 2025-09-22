# api/monitoring.py - API metrics collection
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import time
import psutil
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration', 
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections', 
    'Active HTTP connections'
)

CHAT_REQUESTS = Counter(
    'chat_requests_total', 
    'Total chat requests',
    ['conversation_id']
)

IMAGE_UPLOADS = Counter(
    'image_uploads_total', 
    'Total image uploads'
)

AI_RESPONSE_TIME = Histogram(
    'ai_response_duration_seconds', 
    'AI response generation time'
)

MODEL_PREDICTIONS = Counter(
    'model_predictions_total', 
    'Total model predictions',
    ['model_type']
)

# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self.get_endpoint(request),
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=self.get_endpoint(request)
            ).observe(duration)
            
            return response
            
        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self.get_endpoint(request),
                status=500
            ).inc()
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def get_endpoint(self, request: Request) -> str:
        """Extract endpoint from request"""
        path = request.url.path
        
        # Normalize dynamic paths
        if path.startswith('/conversations/'):
            return '/conversations/{id}'
        elif path.startswith('/images/'):
            return '/images/{id}'
        else:
            return path

def update_system_metrics():
    """Update system metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        SYSTEM_DISK_USAGE.set(disk_percent)
        
    except Exception as e:
        logging.error(f"Error updating system metrics: {e}")

def setup_metrics(app: FastAPI):
    """Setup metrics collection for FastAPI app"""
    
    # Add metrics middleware
    app.add_middleware(MetricsMiddleware)
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        update_system_metrics()
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/app-metrics")
    async def get_app_metrics():
        """Custom application metrics"""
        return {
            "total_requests": REQUEST_COUNT._value._value,
            "active_connections": ACTIVE_CONNECTIONS._value._value,
            "chat_requests": CHAT_REQUESTS._value._value,
            "image_uploads": IMAGE_UPLOADS._value._value,
            "system_metrics": {
                "cpu_usage": SYSTEM_CPU_USAGE._value._value,
                "memory_usage": SYSTEM_MEMORY_USAGE._value._value,
                "disk_usage": SYSTEM_DISK_USAGE._value._value
            }
        }

# Usage in main API file
def record_chat_request(conversation_id: str):
    """Record chat request metric"""
    CHAT_REQUESTS.labels(conversation_id=conversation_id).inc()

def record_image_upload():
    """Record image upload metric"""
    IMAGE_UPLOADS.inc()

def record_ai_response_time(duration: float):
    """Record AI response time"""
    AI_RESPONSE_TIME.observe(duration)

def record_model_prediction(model_type: str):
    """Record model prediction"""
    MODEL_PREDICTIONS.labels(model_type=model_type).inc()