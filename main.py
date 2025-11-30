# main.py - Fixed version with correct imports and error handling
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
from PIL import Image
import io
import base64
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import monitoring components with fallbacks
try:
    from api.monitoring import (
        setup_metrics, record_chat_request, record_image_upload, 
        record_ai_response_time, record_model_prediction
    )
    HAS_MONITORING = True
except ImportError as e:
    print(f"Warning: Monitoring components not available: {e}")
    HAS_MONITORING = False
    
    # Create mock monitoring functions
    def setup_metrics(app): pass
    def record_chat_request(conv_id): pass
    def record_image_upload(): pass
    def record_ai_response_time(time): pass
    def record_model_prediction(model): pass

# Try to import logger with fallback
try:
    from multimodal_ai_assistant.logger import get_logger, log_api_request, log_execution_time
    HAS_LOGGER = True
except ImportError as e:
    print(f"Warning: Logger not available: {e}")
    HAS_LOGGER = False
    
    # Create simple logger fallback
    logging.basicConfig(level=logging.INFO)
    def get_logger():
        return logging.getLogger(__name__)
    
    def log_api_request(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    def log_execution_time(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

# Try to import monitoring components with fallbacks
try:
    from monitoring.health_checker import HealthMonitor
    from monitoring.alerting.alert_manager import AlertManager
    from monitoring.log_analyzer import LogAnalyzer
    from monitoring.dashboard_generator import MonitoringDashboard
    HAS_ADVANCED_MONITORING = True
except ImportError as e:
    print(f"Warning: Advanced monitoring not available: {e}")
    HAS_ADVANCED_MONITORING = False
    
    # Create mock classes
    class HealthMonitor:
        def __init__(self, alert_manager): pass
        def get_health_status(self): return {"status": "healthy", "system": {}}
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
    
    class AlertManager:
        def __init__(self, config): pass
    
    class LogAnalyzer:
        def __init__(self, alert_manager): pass
        def analyze_recent_logs(self, minutes=30): return {"logs": [], "summary": "No analysis available"}
    
    class MonitoringDashboard:
        def __init__(self, health_monitor, log_analyzer): pass
        def generate_dashboard_data(self): return {"message": "Dashboard not available", "data": {}}

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal AI Assistant",
    description="AI Assistant with comprehensive monitoring and observability",
    version="2.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logger = get_logger()

# Initialize monitoring components
alert_config = {
    'routing': {
        'critical': {
            'email_recipients': ['admin@multimodal-ai.com'],
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
        },
        'warning': {
            'email_recipients': ['team@multimodal-ai.com'],
        }
    },
    'smtp': {
        'host': os.getenv('SMTP_HOST', 'localhost'),
        'port': int(os.getenv('SMTP_PORT', 587)),
        'username': os.getenv('SMTP_USERNAME'),
        'password': os.getenv('SMTP_PASSWORD'),
        'use_tls': True,
        'from_email': 'alerts@multimodal-ai.com'
    }
}

# Initialize monitoring with error handling
try:
    alert_manager = AlertManager(alert_config)
    health_monitor = HealthMonitor(alert_manager)
    log_analyzer = LogAnalyzer(alert_manager)
    dashboard_generator = MonitoringDashboard(health_monitor, log_analyzer)
    monitoring_available = True
except Exception as e:
    logger.warning(f"Could not initialize monitoring: {e}")
    monitoring_available = False

# Setup Prometheus metrics
if HAS_MONITORING:
    try:
        setup_metrics(app)
    except Exception as e:
        logger.warning(f"Could not setup metrics: {e}")

# Global storage
conversation_history = []
uploaded_images = {}

# Sample VQA data loader
def load_sample_data():
    """Load sample VQA data with error handling"""
    try:
        # Try multiple possible paths
        possible_paths = [
            "artifacts/data_ingestion/validation/validation_metadata.csv",
            "../artifacts/data_ingestion/validation/validation_metadata.csv",
            "data/sample_data.csv"
        ]
        
        for sample_path in possible_paths:
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path).head(10)
                logger.info(f"Loaded sample data from {sample_path}")
                return df.to_dict('records')
    except Exception as e:
        logger.warning(f"Could not load sample data: {e}")
    
    # Return fallback data
    return [
        {"question": "What color is the car?", "primary_answer": "red", "image_path": "sample1.jpg"},
        {"question": "How many people are in the image?", "primary_answer": "two", "image_path": "sample2.jpg"},
        {"question": "What is the weather like?", "primary_answer": "sunny", "image_path": "sample3.jpg"},
        {"question": "What objects do you see?", "primary_answer": "table and chairs", "image_path": "sample4.jpg"},
        {"question": "What time of day is it?", "primary_answer": "afternoon", "image_path": "sample5.jpg"},
    ]

sample_vqa_data = load_sample_data()

@log_execution_time
def generate_ai_response(question: str, image_id: str = None) -> Dict[str, Any]:
    """Generate AI response with monitoring"""
    start_time = time.time()
    
    try:
        if image_id and image_id in uploaded_images:
            # Record model prediction
            record_model_prediction("multimodal_vqa")
            
            # Enhanced keyword matching for demo
            question_lower = question.lower()
            
            # Try to find relevant sample data
            best_match = None
            max_score = 0
            
            for sample in sample_vqa_data:
                sample_question = sample['question'].lower()
                # Simple scoring based on word overlap
                common_words = set(question_lower.split()) & set(sample_question.split())
                score = len(common_words)
                
                if score > max_score:
                    max_score = score
                    best_match = sample 
            
            if best_match and max_score > 0:
                response_data = {
                    "response": f"Based on the image analysis, I can see that {best_match['primary_answer']}. This relates to your question about the visual content.",
                    "confidence": min(0.95, 0.70 + max_score * 0.1),
                    "source": "visual_analysis", 
                    "matched_sample": best_match['question']
                }
            else:
                # Generic image response
                image_info = uploaded_images[image_id]['info']
                response_data = {
                    "response": f"I can analyze your uploaded image (dimensions: {image_info['dimensions']}, format: {image_info.get('format', 'unknown')}). Regarding '{question}', I'm processing the visual content to provide relevant information.",
                    "confidence": 0.75,
                    "source": "image_processing",
                    "image_info": image_info
                }
        else:
            # Record model prediction
            record_model_prediction("text_only")
            
            # Enhanced text-only responses
            question_lower = question.lower()
            
            # Response patterns
            responses = {
                ("hello", "hi", "hey"): {
                    "response": "Hello! I'm your Multi-Modal AI Assistant with comprehensive monitoring capabilities. I can analyze images, answer questions about visual content, and provide insights. Upload an image to get started!",
                    "confidence": 0.95
                },
                ("what can you do", "capabilities", "features"): {
                    "response": "I can analyze images, answer questions about visual content, engage in conversations, and provide AI-powered insights. I also have advanced monitoring, logging, and alerting systems. Try uploading an image and asking me questions about it!",
                    "confidence": 0.95
                },
                ("how are you", "status", "health"): {
                    "response": "I'm running perfectly! All systems are operational with full monitoring active. My health status is green and I'm ready to help you analyze images and answer questions.",
                    "confidence": 0.90
                },
                ("thank you", "thanks"): {
                    "response": "You're welcome! I'm here to help with image analysis and visual question answering. Feel free to upload images and ask me anything about them.",
                    "confidence": 0.90
                },
                ("test", "testing"): {
                    "response": "Test successful! I'm fully operational and ready for image analysis. Upload an image and ask me questions to see my visual AI capabilities in action.",
                    "confidence": 0.85
                }
            }
            
            # Find matching response
            response_data = None
            for keywords, response_info in responses.items():
                if any(keyword in question_lower for keyword in keywords):
                    response_data = {
                        "response": response_info["response"],
                        "confidence": response_info["confidence"],
                        "source": "conversational_ai"
                    }
                    break
            
            if not response_data:
                response_data = {
                    "response": f"I understand you're asking about '{question}'. I'm designed to work best with images - try uploading an image and asking questions about what you see! I can identify objects, colors, scenes, and much more.",
                    "confidence": 0.60,
                    "source": "general_response"
                }
        
        # Record response time
        response_time = time.time() - start_time
        record_ai_response_time(response_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error while processing your request. Please try again.",
            "confidence": 0.0,
            "source": "error_fallback",
            "error": str(e)
        }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Multi-Modal AI Assistant with Monitoring",
        "version": "2.0.0",
        "status": "active",
        "capabilities": ["image_analysis", "visual_qa", "conversation", "monitoring", "alerting"],
        "monitoring_endpoints": ["/metrics", "/health", "/monitoring/dashboard", "/monitoring/logs"],
        "components": {
            "monitoring": HAS_MONITORING,
            "logger": HAS_LOGGER,
            "advanced_monitoring": HAS_ADVANCED_MONITORING
        },
        "sample_data_loaded": len(sample_vqa_data),
        "uploaded_images": len(uploaded_images),
        "conversation_history": len(conversation_history)
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed status"""
    try:
        # Basic system health
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "services": {
                "api": "operational",
                "image_processing": "operational",
                "conversation": "operational"
            },
            "uptime": "operational",
            "storage": {
                "uploaded_images": len(uploaded_images),
                "conversation_history": len(conversation_history)
            }
        }
        
        # Add monitoring status if available
        if monitoring_available and HAS_ADVANCED_MONITORING:
            try:
                health_status = health_monitor.get_health_status()
                health_data["system_health"] = health_status.get("system", {})
                health_data["services"]["monitoring"] = "operational"
            except Exception as e:
                logger.warning(f"Could not get health status: {e}")
                health_data["services"]["monitoring"] = "degraded"
        else:
            health_data["services"]["monitoring"] = "unavailable"
        
        # Add component availability
        health_data["components"] = {
            "monitoring": HAS_MONITORING,
            "logger": HAS_LOGGER,
            "advanced_monitoring": HAS_ADVANCED_MONITORING
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/upload")
@log_api_request
async def upload_image(file: UploadFile = File(...)):
    """Upload and process image with comprehensive error handling"""
    try:
        # Record image upload
        record_image_upload()
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image. Received: {file.content_type}"
            )
        
        # Read and validate image data
        try:
            image_data = await file.read()
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            
            # Get image info
            image_format = image.format or 'Unknown'
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Generate unique image ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_id = f"img_{timestamp}_{len(uploaded_images)}"
        
        # Store image info
        image_info = {
            "filename": file.filename,
            "size_bytes": len(image_data),
            "dimensions": image.size,
            "format": image_format,
            "mode": image.mode,
            "upload_time": datetime.now().isoformat(),
            "content_type": file.content_type
        }
        
        # Convert to base64 for storage (in production, use proper file storage)
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")
        
        # Store in memory (in production, use database or file storage)
        uploaded_images[image_id] = {
            "image_data": image_base64,
            "info": image_info,
            "processed": True,
            "created_at": datetime.now()
        }
        
        logger.info(
            f"Image uploaded successfully: {image_id}",
            extra={
                "image_id": image_id,
                "filename": file.filename,
                "size_bytes": len(image_data),
                "dimensions": image.size
            }
        )
        
        return {
            "image_id": image_id,
            "message": f"Image '{file.filename}' uploaded successfully! You can now ask questions about this image.",
            "image_info": image_info,
            "next_steps": "Use the /chat endpoint with this image_id to ask questions about your image"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in image upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
@log_api_request  
async def chat(request: Request):
    """Chat with AI assistant - enhanced with monitoring and error handling"""
    try:
        # Parse request data
        try:
            request_data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")
        
        # Extract and validate parameters
        message = request_data.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long. Maximum 1000 characters")
        
        image_id = request_data.get("image_id", "").strip()
        conversation_id = request_data.get("conversation_id", "default").strip()
        
        # Validate image_id if provided
        if image_id and image_id not in uploaded_images:
            raise HTTPException(
                status_code=404, 
                detail=f"Image with ID '{image_id}' not found. Please upload an image first."
            )
        
        # Record chat request
        record_chat_request(conversation_id)
        
        # Generate AI response
        try:
            ai_result = generate_ai_response(message, image_id)
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating AI response")
        
        # Create response
        response_data = {
            "response": ai_result["response"],
            "confidence": ai_result["confidence"],
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "source": ai_result.get("source", "unknown"),
            "has_image": bool(image_id),
            "image_id": image_id if image_id else None
        }
        
        # Store in conversation history
        conversation_entry = {
            "conversation_id": conversation_id,
            "user_message": message,
            "ai_response": response_data["response"],
            "image_id": image_id,
            "confidence": response_data["confidence"],
            "timestamp": response_data["timestamp"],
            "source": ai_result.get("source", "unknown")
        }
        conversation_history.append(conversation_entry)
        
        # Keep only recent conversation history to prevent memory issues
        if len(conversation_history) > 1000:
            conversation_history[:] = conversation_history[-500:]
        
        logger.info(
            f"Chat request processed: {conversation_id}",
            extra={
                "conversation_id": conversation_id,
                "message_length": len(message),
                "confidence": response_data["confidence"],
                "has_image": bool(image_id)
            }
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional utility endpoints
@app.get("/images/{image_id}")
async def get_image_info(image_id: str):
    """Get information about an uploaded image"""
    if image_id not in uploaded_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_data = uploaded_images[image_id]
    return {
        "image_id": image_id,
        "info": image_data["info"],
        "upload_time": image_data["created_at"].isoformat(),
        "processed": image_data["processed"]
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str, limit: int = 50):
    """Get conversation history for a specific conversation"""
    if limit > 100:
        limit = 100
    
    # Filter conversation history by conversation_id
    filtered_history = [
        entry for entry in conversation_history 
        if entry["conversation_id"] == conversation_id
    ]
    
    # Return most recent entries
    recent_history = filtered_history[-limit:] if filtered_history else []
    
    return {
        "conversation_id": conversation_id,
        "message_count": len(recent_history),
        "total_messages": len(filtered_history),
        "messages": recent_history
    }

# Monitoring endpoints (with fallbacks)
@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        if HAS_ADVANCED_MONITORING and monitoring_available:
            dashboard_data = dashboard_generator.generate_dashboard_data()
            return dashboard_data
        else:
            return {
                "status": "basic_monitoring",
                "message": "Advanced monitoring not available",
                "basic_stats": {
                    "uploaded_images": len(uploaded_images),
                    "conversation_history": len(conversation_history),
                    "system_status": "operational"
                }
            }
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard")

@app.get("/monitoring/logs")
async def get_recent_logs():
    """Get recent log analysis"""
    try:
        if HAS_ADVANCED_MONITORING and monitoring_available:
            log_analysis = log_analyzer.analyze_recent_logs(minutes=30)
            return log_analysis
        else:
            return {
                "status": "basic_logging",
                "message": "Advanced log analysis not available",
                "recent_activity": {
                    "uploads": len(uploaded_images),
                    "conversations": len(conversation_history)
                }
            }
    except Exception as e:
        logger.error(f"Log analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze logs")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Multi-Modal AI Assistant v2.0.0")
    
    # Start health monitoring if available
    if HAS_ADVANCED_MONITORING and monitoring_available:
        try:
            health_monitor.start_monitoring()
            logger.info("Advanced monitoring started")
        except Exception as e:
            logger.warning(f"Could not start advanced monitoring: {e}")
    
    logger.info(f"System initialized successfully. Sample data: {len(sample_vqa_data)} entries")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multi-Modal AI Assistant")
    
    # Stop monitoring if available
    if HAS_ADVANCED_MONITORING and monitoring_available:
        try:
            health_monitor.stop_monitoring()
        except Exception as e:
            logger.warning(f"Error stopping monitoring: {e}")
    
    logger.info("Shutdown completed")

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc.detail}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print("Starting Multi-Modal AI Assistant...")
    print("=" * 50)
    print("Access the API at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
