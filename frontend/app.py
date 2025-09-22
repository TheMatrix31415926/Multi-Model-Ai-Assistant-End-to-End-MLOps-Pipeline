"""app.py module"""

# frontend/app.py
import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Multi-Modal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
    .confidence-score {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .upload-area {
        border: 2px dashed #ccc;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_image_id' not in st.session_state:
    st.session_state.current_image_id = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Helper functions
def call_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """Call API endpoint"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("🚫 Cannot connect to API. Please ensure the FastAPI server is running on localhost:8000")
        st.info("Run: `uvicorn api.main:app --reload` in your terminal")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def upload_image(image_file):
    """Upload image to API"""
    files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
    return call_api("upload", "POST", files=files)

def send_message(message: str, image_id: str = None):
    """Send chat message to API"""
    data = {
        "message": message,
        "image_id": image_id,
        "conversation_id": "streamlit_session"
    }
    return call_api("chat", "POST", data)

def display_message(message: dict, is_user: bool = True):
    """Display chat message"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence = message.get('confidence', 0.5)
        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
        
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 AI Assistant:</strong> {message['content']}
            <div class="confidence-score">
                Confidence: <span style="color: {confidence_color};">{confidence:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Modal AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # API Health Check
        if st.button("🔍 Check API Status"):
            health = call_api("health")
            if health:
                st.success("✅ API is healthy!")
                st.json(health)
            else:
                st.error("❌ API is not responding")
        
        st.divider()
        
        # Image Upload
        st.header("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to ask questions about it"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Upload to API
            if st.button("🚀 Process Image"):
                with st.spinner("Uploading image..."):
                    result = upload_image(uploaded_file)
                    if result:
                        st.session_state.current_image_id = result['image_id']
                        st.session_state.uploaded_image = image
                        st.success(f"✅ Image uploaded! ID: {result['image_id']}")
                        st.json(result['image_info'])
                    else:
                        st.error("Failed to upload image")
        
        st.divider()
        
        # Current Image Info
        if st.session_state.current_image_id:
            st.header("🖼️ Current Image")
            st.success(f"ID: {st.session_state.current_image_id}")
            if st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, width=200)
            
            if st.button("🗑️ Clear Image"):
                st.session_state.current_image_id = None
                st.session_state.uploaded_image = None
                st.rerun()
        
        st.divider()
        
        # Statistics
        st.header("📊 Stats")
        if st.button("📈 Get Stats"):
            stats = call_api("stats")
            if stats:
                st.metric("Total Messages", stats.get('total_messages', 0))
                st.metric("Total Images", stats.get('total_images', 0))
                st.metric("Conversations", stats.get('total_conversations', 0))
    
    # Main Chat Interface
    st.header("💬 Chat with AI Assistant")
    
    # Chat History
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for message in st.session_state.conversation_history:
            display_message({"content": message['user']}, is_user=True)
            display_message({
                "content": message['ai'], 
                "confidence": message.get('confidence', 0.5)
            }, is_user=False)
    
    # Chat Input
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me anything about the image or general questions:",
            placeholder="e.g., What do you see in this image? What color is the car?",
            key="chat_input"
        )
    
    with col2:
        send_button = st.button("📤 Send", type="primary")
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.conversation_history.append({
            "user": user_input,
            "ai": "",
            "confidence": 0,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to API
        with st.spinner("🤔 AI is thinking..."):
            response = send_message(user_input, st.session_state.current_image_id)
            
            if response:
                # Update conversation history
                st.session_state.conversation_history[-1].update({
                    "ai": response['response'],
                    "confidence": response['confidence']
                })
                
                # Clear input and rerun
                st.rerun()
            else:
                st.error("Failed to get AI response")
    
    # Quick Actions
    st.divider()
    st.header("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("👋 Say Hello"):
            if st.session_state.current_image_id:
                quick_message = "Hello! Can you tell me what you see in this image?"
            else:
                quick_message = "Hello! How can you help me today?"
            
            st.session_state.conversation_history.append({"user": quick_message, "ai": "", "confidence": 0})
            response = send_message(quick_message, st.session_state.current_image_id)
            if response:
                st.session_state.conversation_history[-1].update({
                    "ai": response['response'],
                    "confidence": response['confidence']
                })
                st.rerun()
    
    with col2:
        if st.button("❓ What can you do?"):
            quick_message = "What can you do?"
            st.session_state.conversation_history.append({"user": quick_message, "ai": "", "confidence": 0})
            response = send_message(quick_message, st.session_state.current_image_id)
            if response:
                st.session_state.conversation_history[-1].update({
                    "ai": response['response'],
                    "confidence": response['confidence']
                })
                st.rerun()
    
    with col3:
        if st.button("🎨 Describe Image") and st.session_state.current_image_id:
            quick_message = "Can you describe what you see in this image in detail?"
            st.session_state.conversation_history.append({"user": quick_message, "ai": "", "confidence": 0})
            response = send_message(quick_message, st.session_state.current_image_id)
            if response:
                st.session_state.conversation_history[-1].update({
                    "ai": response['response'],
                    "confidence": response['confidence']
                })
                st.rerun()
    
    with col4:
        if st.button("🧹 Clear Chat"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Demo Instructions
    if not st.session_state.conversation_history:
        st.info("""
        ### 🎯 How to use this Multi-Modal AI Assistant:
        
        1. **📷 Upload an Image**: Use the sidebar to upload an image
        2. **🚀 Process Image**: Click "Process Image" to analyze it
        3. **💬 Ask Questions**: Type questions about the image or general topics
        4. **🔍 Explore**: Try the quick action buttons for common queries
        
        ### 💡 Example questions:
        - "What do you see in this image?"
        - "What color is the car?"
        - "How many people are there?"
        - "What's the weather like?"
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Multi-Modal AI Assistant v1.0 | Built with FastAPI + Streamlit | 
    <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
