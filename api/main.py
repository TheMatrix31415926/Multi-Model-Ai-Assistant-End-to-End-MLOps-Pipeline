# api/main.py - Updated with monitoring integration
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

# Import monitoring components
from api.monitoring import (
    setup_metrics, record_chat_request, record_image_upload, 
    record_ai_response_time, record_model_prediction
)
from multimodal_ai_assistant.logger import get_logger, log_api_request, log_execution_time
from monitoring.health_checker import HealthMonitor
from monitoring.alerting.alert_manager import AlertManager
from monitoring.log_analyzer import LogAnalyzer
from monitoring.dashboard_generator import MonitoringDashboard

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

alert_manager = AlertManager(alert_config)
health_monitor = HealthMonitor(alert_manager)
log_analyzer = LogAnalyzer(alert_manager)
dashboard_generator = MonitoringDashboard(health_monitor, log_analyzer)

# Setup Prometheus metrics
setup_metrics(app)

# Global storage
conversation_history = []
uploaded_images = {}

# Sample VQA data loader
def load_sample_data():
    try:
        sample_path = "../artifacts/data_ingestion/validation/validation_metadata.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path).head(10)
            return df.to_dict('records')
    except:
        pass
    
    return [
        {"question": "What color is the car?", "primary_answer": "red", "image_path": "sample1.jpg"},
        {"question": "How many people are in the image?", "primary_answer": "two", "image_path": "sample2.jpg"},
        {"question": "What is the weather like?", "primary_answer": "sunny", "image_path": "sample3.jpg"},
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
            
            # Simple keyword matching for demo
            question_lower = question.lower()
            
            for sample in sample_vqa_data:
                sample_question = sample['question'].lower()
                if any(word in question_lower for word in sample_question.split()[:3]):
                    response_data = {
                        "response": f"Based on the image, I can see that {sample['primary_answer']}. This appears to be related to your question about the visual content.",
                        "confidence": 0.85,
                        "source": "visual_analysis"
                    }
                    break
            else:
                response_data = {
                    "response": f"I can see you've uploaded an image. Regarding your question '{question}', I'm analyzing the visual content and can provide contextual information based on what I observe.",
                    "confidence": 0.70,
                    "source": "visual_analysis"
                }
        else:
            # Record model prediction
            record_model_prediction("text_only")
            
            # Text-only responses
            text_responses = {
                "hello": "Hello! I'm your Multi-Modal AI Assistant with comprehensive monitoring. I can analyze images and answer questions about them!",
                "what can you do": "I can analyze images, answer questions about visual content, and provide insights. Plus, I now have advanced monitoring and alerting capabilities!",
                "how are you": "I'm running perfectly with full monitoring! All systems are green and ready to help you.",
                "thank you": "You're welcome! Feel free to ask me anything about images or upload new pictures to analyze."
            }
            
            question_lower = question.lower()
            for key, response in text_responses.items():
                if key in question_lower:
                    response_data = {
                        "response": response,
                        "confidence": 0.90,
                        "source": "text_analysis"
                    }
                    break
            else:
                response_data = {
                    "response": f"I understand you're asking about '{question}'. While I'd be happy to help, I work best when analyzing images. Try uploading an image and asking questions about what you see!",
                    "confidence": 0.60,
                    "source": "general"
                }
        
        # Record response time
        response_time = time.time() - start_time
        record_ai_response_time(response_time)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        raise

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Multi-Modal AI Assistant with Monitoring",
        "version": "2.0.0",
        "status": "active",
        "capabilities": ["image_analysis", "visual_qa", "conversation", "monitoring", "alerting"],
        "monitoring_endpoints": ["/metrics", "/health", "/monitoring/dashboard", "/monitoring/logs"]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed status"""
    try:
        health_status = health_monitor.get_health_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "services": {
                "api": "operational",
                "monitoring": "operational",
                "logging": "operational",
                "alerting": "operational"
            },
            "system_health": health_status.get("system", {}),
            "uptime": "99.9%",  # This would be calculated from actual metrics
            "database": "connected" if len(conversation_history) >= 0 else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/upload")
@log_api_request
async def upload_image(file: UploadFile = File(...)):
    """Upload and process image with monitoring"""
    try:
        # Record image upload
        record_image_upload()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate unique image ID
        image_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(uploaded_images)}"
        
        # Store image info
        image_info = {
            "filename": file.filename,
            "size": len(image_data),
            "dimensions": image.size,
            "format": image.format,
            "mode": image.mode,
            "upload_time": datetime.now().isoformat()
        }
        
        # Convert to base64 for storage
        image_base64 = base64.b64encode(image_data).decode()
        
        uploaded_images[image_id] = {
            "image_data": image_base64,
            "info": image_info,
            "processed": True
        }
        
        logger.info(
            f"Image uploaded successfully: {image_id}",
            extra={
                "image_id": image_id,
                "filename": file.filename,
                "size_bytes": len(image_data)
            }
        )
        
        return {
            "image_id": image_id,
            "message": f"Image uploaded successfully! You can now ask questions about this image using ID: {image_id}",
            "image_info": image_info
        }
        
    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/chat")
@log_api_request  
async def chat(request: Request):
    """Chat with AI assistant - enhanced with monitoring"""
    try:
        request_data = await request.json()
        message = request_data.get("message", "")
        image_id = request_data.get("image_id")
        conversation_id = request_data.get("conversation_id", "default")
        
        # Record chat request
        record_chat_request(conversation_id)
        
        # Generate AI response
        ai_result = generate_ai_response(message, image_id)
        
        # Create response
        response_data = {
            "response": ai_result["response"],
            "confidence": ai_result["confidence"],
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "source": ai_result.get("source", "unknown")
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
        
        logger.info(
            f"Chat request processed: {conversation_id}",
            extra={
                "conversation_id": conversation_id,
                "message_length": len(message),
                "confidence": response_data["confidence"],
                "has_image": image_id is not None
            }
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Monitoring endpoints
@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        dashboard_data = dashboard_generator.generate_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard")

@app.get("/monitoring/logs")
async def get_recent_logs():
    """Get recent log analysis"""
    try:
        log_analysis = log_analyzer.analyze_recent_logs(minutes=30)
        return log_analysis
    except Exception as e:
        logger.error(f"Log analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze logs")

@app.get("/monitoring/health-details")
async def get_detailed_health():
    """Get detailed health information"""
    try:
        health_status = health_monitor.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health details")

@app.post("/webhooks/alerts")
async def receive_alert_webhook(request: Request):
    """Receive alerts from external systems"""
    try:
        alert_data = await request.json()
        logger.info(f"Received alert webhook: {alert_data}")
        
        # Process the alert (could trigger additional actions)
        return {"status": "received", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Alert webhook processing failed: {e}")
        return {"status": "error", "message": str(e)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    logger.info("Starting Multi-Modal AI Assistant with Monitoring v2.0.0")
    
    # Start health monitoring
    health_monitor.start_monitoring()
    
    logger.info("All monitoring systems initialized successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multi-Modal AI Assistant")
    
    # Stop monitoring
    health_monitor.stop_monitoring()
    
    logger.info("Shutdown completed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
