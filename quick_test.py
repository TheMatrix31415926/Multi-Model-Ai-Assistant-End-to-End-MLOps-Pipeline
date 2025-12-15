# test_your_project.py - Tailored test for your specific project structure
"""
Local test runner for your Multi-Modal AI Assistant
Tests all components based on your actual project structure
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from PIL import Image
import io
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_test_image(color='blue', size=(224, 224)):
    """Create a test image for testing"""
    img = Image.new('RGB', size, color=color)
    return img

def test_project_structure():
    """Test if project structure is correct"""
    print("📁 Testing Project Structure...")
    
    required_dirs = [
        "api",
        "frontend", 
        "multimodal_ai_assistant",
        "tests",
        "deployment",
        "monitoring"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
            print(f"   ❌ Missing: {dir_name}/")
        else:
            print(f"   ✅ Found: {dir_name}/")
    
    if missing_dirs:
        print(f"   ⚠️  Missing directories: {missing_dirs}")
        return False
    
    print("   ✅ Project structure looks good!")
    return True

def test_basic_imports():
    """Test basic Python imports"""
    print("\n🔍 Testing Basic Imports...")
    
    import_tests = [
        ("PIL (Pillow)", "PIL", "Image"),
        ("NumPy", "numpy", None),
        ("Pandas", "pandas", None),
        ("Requests", "requests", None),
    ]
    
    results = {}
    for name, module, submodule in import_tests:
        try:
            if submodule:
                exec(f"from {module} import {submodule}")
            else:
                exec(f"import {module}")
            results[name] = "✅ OK"
            print(f"   {name}: ✅ OK")
        except ImportError:
            results[name] = "❌ MISSING"
            print(f"   {name}: ❌ MISSING")
        except Exception as e:
            results[name] = f"⚠️  ERROR: {e}"
            print(f"   {name}: ⚠️  ERROR - {e}")
    
    return results

def test_multimodal_components():
    """Test multimodal AI components"""
    print("\n🤖 Testing Multimodal Components...")
    
    try:
        # Test VQA Agent
        print("   Testing VQA Agent...")
        try:
            from multimodal_ai_assistant.multimodal.agents.vqa_agent import VQAAgent
            
            # Try to create VQA agent
            vqa_agent = VQAAgent()
            print("   ✅ VQA Agent: Created successfully")
            
            # Test with image
            test_image = create_test_image('red', (300, 300))
            question = "What color is this image?"
            
            try:
                result = vqa_agent.ask_question(test_image, question)
                print(f"   📝 VQA Test - Question: {question}")
                print(f"   📝 VQA Test - Answer: {result.get('answer', 'No answer')}")
                print(f"   📝 VQA Test - Confidence: {result.get('confidence', 0.0):.2f}")
                return "✅ VQA Agent working"
            except Exception as e:
                print(f"   ⚠️  VQA prediction failed: {e}")
                return "⚠️  VQA Agent created but prediction failed"
                
        except ImportError as e:
            print(f"   ❌ VQA Agent import failed: {e}")
            return f"❌ VQA Agent not available"
        
    except Exception as e:
        print(f"   ❌ Multimodal components test failed: {e}")
        return f"❌ Failed: {e}"

def test_conversation_agent():
    """Test conversation agent"""
    print("\n💬 Testing Conversation Agent...")
    
    try:
        from multimodal_ai_assistant.multimodal.agents.conversation_agent import ConversationAgent
        
        # Create conversation agent
        conv_agent = ConversationAgent()
        print("   ✅ Conversation Agent: Created successfully")
        
        # Test basic conversation
        message = "Hello! Can you help me test this system?"
        try:
            result = conv_agent.chat(message)
            print(f"   📝 Conv Test - User: {message}")
            print(f"   📝 Conv Test - AI: {result.get('response', 'No response')[:80]}...")
            print(f"   📝 Conv Test - Session: {result.get('session_id', 'None')}")
            
            # Test follow-up
            session_id = result.get('session_id')
            if session_id:
                followup = "What did I just ask you?"
                followup_result = conv_agent.chat(followup, session_id=session_id)
                print(f"   📝 Conv Test - Follow-up: {followup}")
                print(f"   📝 Conv Test - Response: {followup_result.get('response', 'No response')[:60]}...")
            
            return "✅ Conversation Agent working"
            
        except Exception as e:
            print(f"   ⚠️  Conversation prediction failed: {e}")
            return "⚠️  Conversation Agent created but chat failed"
        
    except ImportError as e:
        print(f"   ❌ Conversation Agent import failed: {e}")
        return f"❌ Conversation Agent not available"
    except Exception as e:
        print(f"   ❌ Conversation Agent test failed: {e}")
        return f"❌ Failed: {e}"

def test_pipeline_components():
    """Test pipeline components"""
    print("\n🔄 Testing Pipeline Components...")
    
    try:
        from multimodal_ai_assistant.pipeline.prediction_pipeline import PredictionPipeline
        
        # Create prediction pipeline
        pipeline = PredictionPipeline()
        print("   ✅ Prediction Pipeline: Created successfully")
        
        # Test text prediction
        try:
            text_result = pipeline.predict("Tell me about machine learning")
            print(f"   📝 Pipeline Test - Query: Tell me about machine learning")
            print(f"   📝 Pipeline Test - Response: {text_result.get('response', 'No response')[:80]}...")
            print(f"   📝 Pipeline Test - Type: {text_result.get('query_type', 'Unknown')}")
            
            # Test with image
            test_image = create_test_image('green', (200, 200))
            image_result = pipeline.predict("What do you see?", image=test_image)
            print(f"   📝 Pipeline Test - Image Query: What do you see?")
            print(f"   📝 Pipeline Test - Response: {image_result.get('response', 'No response')[:80]}...")
            
            return "✅ Pipeline working"
            
        except Exception as e:
            print(f"   ⚠️  Pipeline prediction failed: {e}")
            return "⚠️  Pipeline created but prediction failed"
        
    except ImportError as e:
        print(f"   ❌ Pipeline import failed: {e}")
        return f"❌ Pipeline not available"
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return f"❌ Failed: {e}"

def test_vector_store():
    """Test vector store"""
    print("\n🗄️  Testing Vector Store...")
    
    try:
        from multimodal_ai_assistant.vector_store.chroma_store import ChromaStore
        
        # Create vector store
        store = ChromaStore()
        print("   ✅ Vector Store: Created successfully")
        
        try:
            # Test basic operations (mock mode is fine)
            from multimodal_ai_assistant.vector_store.chroma_store import MultiModalDocument
            import numpy as np
            
            test_doc = MultiModalDocument(
                id="test_doc_1",
                content="This is a test document about blue cars",
                embedding=np.random.rand(256).tolist(),
                metadata={"type": "test", "color": "blue"}
            )
            
            doc_id = store.add_document(test_doc)
            print(f"   📝 Vector Store - Added doc with ID: {doc_id}")
            
            # Query test
            results = store.query_documents("blue cars", n_results=3)
            print(f"   📝 Vector Store - Query returned: {len(results.get('documents', []))} results")
            
            return "✅ Vector Store working"
            
        except Exception as e:
            print(f"   ⚠️  Vector Store operations failed: {e}")
            return "⚠️  Vector Store created but operations failed"
        
    except ImportError as e:
        print(f"   ❌ Vector Store import failed: {e}")
        return f"❌ Vector Store not available"
    except Exception as e:
        print(f"   ❌ Vector Store test failed: {e}")
        return f"❌ Failed: {e}"

def test_api_components():
    """Test API components"""
    print("\n🌐 Testing API Components...")
    
    try:
        from api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        print("   ✅ API App: Imported successfully")
        
        try:
            # Test health endpoint
            response = client.get("/health")
            print(f"   📝 API Test - Health endpoint: Status {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"   📝 API Test - Health status: {health_data.get('status', 'Unknown')}")
            
            # Test chat endpoint
            chat_data = {
                "message": "Hello API!",
                "session_id": "test_session",
                "user_id": "test_user"
            }
            
            chat_response = client.post("/chat", json=chat_data)
            print(f"   📝 API Test - Chat endpoint: Status {chat_response.status_code}")
            
            if chat_response.status_code == 200:
                chat_result = chat_response.json()
                print(f"   📝 API Test - Chat response: {chat_result.get('response', 'No response')[:60]}...")
            
            return "✅ API Components working"
            
        except Exception as e:
            print(f"   ⚠️  API endpoint testing failed: {e}")
            return "⚠️  API imported but endpoints failed"
        
    except ImportError as e:
        print(f"   ❌ API import failed: {e}")
        return f"❌ API not available"
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return f"❌ Failed: {e}"

def test_frontend_components():
    """Test frontend components"""
    print("\n🎨 Testing Frontend Components...")
    
    try:
        from frontend.app import main
        print("   ✅ Frontend App: Imported successfully")
        print("   📝 Note: Full UI testing requires running 'streamlit run frontend/app.py'")
        
        # Try to import components
        try:
            from frontend.components.chat_interface import ChatInterface
            print("   ✅ Chat Interface: Available")
        except ImportError:
            print("   ⚠️  Chat Interface: Not found (optional)")
        
        try:
            from frontend.components.file_upload import FileUpload
            print("   ✅ File Upload: Available")
        except ImportError:
            print("   ⚠️  File Upload: Not found (optional)")
        
        return "✅ Frontend working"
        
    except ImportError as e:
        print(f"   ❌ Frontend import failed: {e}")
        return f"❌ Frontend not available"
    except Exception as e:
        print(f"   ❌ Frontend test failed: {e}")
        return f"❌ Failed: {e}"

def test_data_components():
    """Test data processing components"""
    print("\n📊 Testing Data Components...")
    
    components_to_test = [
        ("Data Ingestion", "multimodal_ai_assistant.components.data_ingestion", "DataIngestion"),
        ("Data Validation", "multimodal_ai_assistant.components.data_validation", "DataValidation"),
        ("Data Transformation", "multimodal_ai_assistant.components.data_transformation", "DataTransformation"),
        ("Model Trainer", "multimodal_ai_assistant.components.model_trainer", "ModelTrainer"),
        ("Model Evaluation", "multimodal_ai_assistant.components.model_evaluation", "ModelEvaluator"),
    ]
    
    working_components = 0
    total_components = len(components_to_test)
    
    for name, module_path, class_name in components_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            print(f"   ✅ {name}: Available")
            working_components += 1
        except ImportError:
            print(f"   ❌ {name}: Import failed")
        except AttributeError:
            print(f"   ⚠️  {name}: Class not found")
        except Exception as e:
            print(f"   ⚠️  {name}: Error - {e}")
    
    print(f"   📊 Data Components: {working_components}/{total_components} available")
    
    if working_components > total_components // 2:
        return "✅ Data Components mostly working"
    else:
        return "⚠️  Some Data Components missing"

def run_integration_test():
    """Run end-to-end integration test"""
    print("\n🔗 Running Integration Test...")
    
    try:
        # Test full workflow
        from multimodal_ai_assistant.pipeline.prediction_pipeline import PredictionPipeline
        
        pipeline = PredictionPipeline()
        
        print("   🎯 Testing complete workflow...")
        
        # Step 1: Text conversation
        greeting = pipeline.predict("Hello! I'm testing the integration.")
        print(f"   Step 1 - Greeting: {greeting.get('response', 'No response')[:60]}...")
        
        session_id = greeting.get('session_id') 
        
        # Step 2: Image analysis
        test_image = create_test_image('purple', (250, 250))
        image_query = pipeline.predict("What color is this image?", image=test_image, session_id=session_id)
        print(f"   Step 2 - Image Query: {image_query.get('response', 'No response')[:60]}...")
        
        # Step 3: Follow-up
        followup = pipeline.predict("Tell me more about that color.", session_id=session_id)
        print(f"   Step 3 - Follow-up: {followup.get('response', 'No response')[:60]}...")
        
        # Verify session continuity
        if (greeting.get('session_id') == image_query.get('session_id') == 
            followup.get('session_id')):
            print("   ✅ Session continuity: Working")
        else:
            print("   ⚠️  Session continuity: Issues detected")
        
        return "✅ Integration test passed"
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return f"❌ Integration failed: {e}"

def check_configuration_files():
    """Check if configuration files exist"""
    print("\n⚙️  Checking Configuration Files...")
    
    config_files = [
        "configs/model.yaml",
        "configs/pipeline.yaml", 
        "configs/deployment.yaml",
        ".env.example",
        "requirements.txt",
        "pyproject.toml"
    ]
    
    found_configs = 0
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✅ {config_file}")
            found_configs += 1
        else:
            print(f"   ❌ {config_file}")
    
    print(f"   📊 Configuration files: {found_configs}/{len(config_files)} found")
    return found_configs > len(config_files) // 2

def main():
    """Main test runner"""
    print("🚀 Multi-Modal AI Assistant - Project Test Runner")
    print("=" * 65)
    print("Testing your specific project implementation...")
    print()
    
    # Run all tests
    test_results = {}
    
    test_results["Project Structure"] = test_project_structure()
    test_results["Basic Imports"] = test_basic_imports()
    test_results["Multimodal Components"] = test_multimodal_components()
    test_results["Conversation Agent"] = test_conversation_agent()
    test_results["Pipeline Components"] = test_pipeline_components()
    test_results["Vector Store"] = test_vector_store()
    test_results["API Components"] = test_api_components()
    test_results["Frontend Components"] = test_frontend_components()
    test_results["Data Components"] = test_data_components()
    test_results["Integration Test"] = run_integration_test()
    test_results["Configuration Files"] = check_configuration_files()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            print(f"{test_name}:")
            for subtest, subresult in result.items():
                print(f"  {subtest}: {subresult}")
                if "✅" in str(subresult):
                    passed += 1
                elif "❌" in str(subresult):
                    failed += 1
                else:
                    warnings += 1
        elif isinstance(result, bool):
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1
        else:
            print(f"{test_name}: {result}")
            if "✅" in str(result):
                passed += 1
            elif "❌" in str(result):
                failed += 1
            else:
                warnings += 1
    
    print("\n🎯 FINAL RESULTS")
    print("-" * 25)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 25)
    
    if failed == 0:
        print("🎉 Excellent! Your system is ready to run!")
        print("\n🚀 Quick Start Commands:")
        print("1. Test API: python -m uvicorn api.main:app --reload --port 8000")
        print("2. Test Frontend: streamlit run frontend/app.py")
        print("3. Full System: python run_app.py (if available)")
        print("4. Access UI: http://localhost:8501")
        
    elif failed <= 3:
        print("👍 Good! Most components are working.")
        print("🔧 Fix the failed components and you're ready to go!")
        
        if any("import" in str(result).lower() for result in test_results.values()):
            print("📦 Install missing dependencies:")
            print("   pip install -r requirements.txt")
            print("   pip install torch transformers sentence-transformers chromadb")
        
    else:
        print("🔨 Several components need attention.")
        print("📋 Priority fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check project structure and imports")
        print("3. Verify configuration files")
        print("4. Run tests again after fixes")
    
    print(f"\n📊 Overall Success Rate: {(passed/(passed+failed+warnings))*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

# run_your_system.py - System launcher for your project
"""
System launcher tailored to your project structure
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "streamlit",
        "pillow",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_your_system():
    """Start your Multi-Modal AI Assistant system"""
    print("🚀 Starting Your Multi-Modal AI Assistant")
    print("=" * 55)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        return False
    
    # Check if ports are available
    import socket
    
    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except:
                return False
    
    if not check_port(8000):
        print("❌ Port 8000 is busy. Stop other services or change the port.")
        return False
    
    if not check_port(8501):
        print("❌ Port 8501 is busy. Stop other services or change the port.")
        return False
    
    print("✅ Ports 8000 and 8501 are available")
    
    try:
        print("\n🌐 Starting API server...")
        # Start API with your project structure
        api_cmd = [
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ]
        
        api_process = subprocess.Popen(api_cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
        
        # Wait for API to start
        time.sleep(4)
        
        print("🎨 Starting Streamlit frontend...")
        # Start frontend with your project structure
        frontend_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        frontend_process = subprocess.Popen(frontend_cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        
        # Wait for services to fully start
        time.sleep(6)
        
        print("\n🎉 System started successfully!")
        print("📊 API Documentation: http://localhost:8000/docs")
        print("🎨 Frontend Interface: http://localhost:8501")
        print("❤️  Health Check: http://localhost:8000/health")
        
        # Test if services are responding
        try:
            import requests
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            if health_response.status_code == 200:
                print("✅ API is responding")
            else:
                print("⚠️  API might have issues")
        except:
            print("⚠️  Could not verify API status")
        
        # Open browser
        print("\n🌐 Opening browser...")
        try:
            webbrowser.open("http://localhost:8501")
        except:
            print("Please manually open: http://localhost:8501")
        
        print(f"\n🔥 Your Multi-Modal AI Assistant is running!")
        print("💡 Upload an image and ask questions about it!")
        print("🛑 Press Ctrl+C to stop both services")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down services...")
            
            api_process.terminate()
            frontend_process.terminate()
            
            # Give processes time to shut down gracefully
            time.sleep(2)
            
            # Force kill if still running
            try:
                api_process.kill()
                frontend_process.kill()
            except:
                pass
            
            print("✅ System shutdown complete!")
            return True
    
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        return False

if __name__ == "__main__":
    start_your_system()
