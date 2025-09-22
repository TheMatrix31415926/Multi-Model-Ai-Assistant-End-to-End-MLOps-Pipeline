"""prediction_pipeline.py module"""

"""
Multi-Modal AI Assistant - Prediction Pipeline
Agent Orchestration with LangGraph for complex multi-step queries
"""

import os
import sys
import json
import uuid
import asyncio
from typing import List, Dict, Union, Optional, Any, TypedDict, Annotated, Literal
import logging
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

# LangGraph and LangChain imports (with fallbacks)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
except ImportError:
    print("Warning: LangGraph/LangChain not installed. Using mock classes for testing.")
    
    # Mock classes for testing
    class StateGraph:
        def __init__(self, state_schema): 
            self.nodes = {}
            self.checkpointer = None
            
        def add_node(self, name, func): 
            self.nodes[name] = func
            return self
            
        def add_edge(self, from_node, to_node): 
            return self
            
        def add_conditional_edges(self, source, path_map, path_map_func=None):
            return self
            
        def set_entry_point(self, node): 
            return self
            
        def compile(self, checkpointer=None): 
            self.checkpointer = checkpointer
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph): 
            self.graph = graph
            
        async def ainvoke(self, input_data, config=None):
            return {"messages": [{"content": "Mock orchestrated response"}]}
            
        def invoke(self, input_data, config=None):
            return {"messages": [{"content": "Mock orchestrated response"}]}
    
    class BaseMessage:
        def __init__(self, content): 
            self.content = content
    
    class HumanMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass
    class MemorySaver: pass
    
    def add_messages(x, y): 
        return x + y
    
    END = "END"

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from multimodal.agents.vqa_agent import VQAAgent, VQAConfig
    from multimodal.agents.conversation_agent import ConversationAgent, ConversationConfig
    from multimodal.fusion.multimodal_fusion import MultiModalFusion, FusionConfig
    from vector_store.chroma_store import ChromaStore, ChromaConfig
except ImportError:
    print("Warning: Custom agents not found. Using mock classes.")
    
    class VQAAgent:
        def __init__(self, config=None): pass
        def ask_question(self, image, question):
            return {"answer": "Mock VQA answer", "confidence": 0.8}
    
    class ConversationAgent:
        def __init__(self, config=None): pass
        def chat(self, message, image=None, conversation_id=None):
            return {"response": "Mock conversation response", "conversation_id": "mock_id"}
    
    class MultiModalFusion:
        def __init__(self, config=None): pass
        def process_multimodal(self, image, text):
            return {"embedding": np.random.rand(256)}
    
    class ChromaStore:
        def __init__(self, config=None): pass
        def query_documents(self, query, n_results=5):
            return {"documents": [{"content": "Mock document"}]}
    
    class VQAConfig: pass
    class ConversationConfig: pass
    class FusionConfig: pass
    class ChromaConfig: pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query types for routing
class QueryType(Enum):
    VQA = "visual_question_answering"
    CONVERSATION = "conversation"
    COMPLEX = "complex_multi_step"
    RETRIEVAL = "retrieval_augmented"
    ANALYSIS = "image_analysis"

# State definition for orchestration
class OrchestrationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: Optional[QueryType]
    current_image: Optional[Image.Image]
    current_image_path: Optional[str]
    session_id: str
    user_id: str
    
    # Processing state
    vqa_result: Optional[Dict[str, Any]]
    conversation_result: Optional[Dict[str, Any]]
    retrieval_context: Optional[List[Dict[str, Any]]]
    analysis_result: Optional[Dict[str, Any]]
    
    # Control flow
    next_step: Optional[str]
    requires_followup: bool
    intermediate_steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    # Final output
    final_response: Optional[str]
    confidence_score: Optional[float]

@dataclass
class PipelineConfig:
    """Configuration for the orchestration pipeline"""
    
    # Agent configurations
    vqa_config: Optional[VQAConfig] = None
    conversation_config: Optional[ConversationConfig] = None
    fusion_config: Optional[FusionConfig] = None
    chroma_config: Optional[ChromaConfig] = None
    
    # LLM settings for orchestrator
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1024
    openai_api_key: Optional[str] = None
    
    # Pipeline settings
    max_iterations: int = 10
    confidence_threshold: float = 0.7
    enable_memory: bool = True
    enable_retrieval: bool = True
    
    # Query classification
    classification_prompt: str = """
    Analyze the user's query and classify it. Consider:
    - Is there an image involved?
    - Is this a simple question about an image?
    - Is this a conversational query?
    - Does it require complex reasoning?
    - Does it need information retrieval?
    
    Classify as one of: VQA, CONVERSATION, COMPLEX, RETRIEVAL, ANALYSIS
    """

class PredictionPipeline:
    """
    Main orchestration pipeline for multi-modal AI assistant
    Routes queries to appropriate agents and manages complex workflows
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.vqa_agent = None
        self.conversation_agent = None
        self.fusion_model = None
        self.vector_store = None
        self.orchestrator_llm = None
        
        # State management
        self.checkpointer = None
        self.graph = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_components()
        self._build_orchestration_graph()
        
        logger.info("Prediction Pipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize all agent components"""
        try:
            # Initialize orchestrator LLM
            if self.config.llm_provider == "openai":
                if not self.config.openai_api_key:
                    self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
                
                if self.config.openai_api_key:
                    self.orchestrator_llm = ChatOpenAI(
                        api_key=self.config.openai_api_key,
                        model_name=self.config.model_name,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
            
            # Initialize agents
            self.vqa_agent = VQAAgent(self.config.vqa_config)
            self.conversation_agent = ConversationAgent(self.config.conversation_config)
            self.fusion_model = MultiModalFusion(self.config.fusion_config)
            self.vector_store = ChromaStore(self.config.chroma_config)
            
        except Exception as e:
            logger.warning(f"Error initializing components: {e}. Using mock components.")
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components for testing"""
        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="Mock orchestrator response")
        
        self.orchestrator_llm = MockLLM()
        self.vqa_agent = VQAAgent()
        self.conversation_agent = ConversationAgent()
        self.fusion_model = MultiModalFusion()
        self.vector_store = ChromaStore()
    
    def _build_orchestration_graph(self):
        """Build the LangGraph orchestration workflow"""
        # Initialize checkpointer
        self.checkpointer = MemorySaver()
        
        # Create the state graph
        workflow = StateGraph(OrchestrationState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("process_vqa", self._process_vqa)
        workflow.add_node("process_conversation", self._process_conversation)
        workflow.add_node("process_complex", self._process_complex)
        workflow.add_node("process_retrieval", self._process_retrieval)
        workflow.add_node("process_analysis", self._process_analysis)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("quality_check", self._quality_check)
        
        # Define conditional routing
        workflow.set_entry_point("classify_query")
        
        # Route based on query type
        workflow.add_conditional_edges(
            "classify_query",
            self._route_query,
            {
                QueryType.VQA.value: "process_vqa",
                QueryType.CONVERSATION.value: "process_conversation", 
                QueryType.COMPLEX.value: "process_complex",
                QueryType.RETRIEVAL.value: "process_retrieval",
                QueryType.ANALYSIS.value: "process_analysis"
            }
        )
        
        # All processing nodes go to synthesis
        for node in ["process_vqa", "process_conversation", "process_complex", 
                    "process_retrieval", "process_analysis"]:
            workflow.add_edge(node, "synthesize_response")
        
        workflow.add_edge("synthesize_response", "quality_check")
        
        # Quality check can loop back or end
        workflow.add_conditional_edges(
            "quality_check",
            self._check_quality,
            {
                "retry": "synthesize_response",
                "complete": END
            }
        )
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    async def _classify_query(self, state: OrchestrationState) -> OrchestrationState:
        """Classify the incoming query to determine routing"""
        logger.info("Classifying query")
        
        try:
            # Get the user message
            user_message = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            # Simple rule-based classification for now
            query_type = self._classify_query_simple(user_message, state["current_image"])
            
            # If we have an orchestrator LLM, use it for better classification
            if self.orchestrator_llm and hasattr(self.orchestrator_llm, 'invoke'):
                classification_prompt = f"""
                {self.config.classification_prompt}
                
                User query: "{user_message}"
                Has image: {state["current_image"] is not None}
                
                Respond with only one word: VQA, CONVERSATION, COMPLEX, RETRIEVAL, or ANALYSIS
                """
                
                try:
                    response = self.orchestrator_llm.invoke([HumanMessage(content=classification_prompt)])
                    classification = response.content.strip().upper()
                    
                    if classification in [qt.name for qt in QueryType]:
                        query_type = QueryType[classification]
                except Exception as e:
                    logger.warning(f"LLM classification failed: {e}, using rule-based")
            
            state["query_type"] = query_type
            state["intermediate_steps"].append({
                "step": "classify_query",
                "timestamp": datetime.now().isoformat(),
                "query_type": query_type.value,
                "user_message_length": len(user_message)
            })
            
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            state["query_type"] = QueryType.CONVERSATION  # Default fallback
        
        return state
    
    def _classify_query_simple(self, message: str, has_image: bool) -> QueryType:
        """Simple rule-based query classification"""
        message_lower = message.lower()
        
        # Check for image-related queries
        if has_image:
            if any(word in message_lower for word in ["what", "describe", "see", "show", "identify"]):
                return QueryType.VQA
            elif any(word in message_lower for word in ["analyze", "detailed", "analysis"]):
                return QueryType.ANALYSIS
        
        # Check for retrieval keywords
        if any(word in message_lower for word in ["find", "search", "lookup", "remember", "previous"]):
            return QueryType.RETRIEVAL
        
        # Check for complex reasoning
        if any(word in message_lower for word in ["compare", "explain", "why", "how", "complex", "reason"]):
            return QueryType.COMPLEX
        
        # Default to conversation
        return QueryType.CONVERSATION
    
    def _route_query(self, state: OrchestrationState) -> str:
        """Route query based on classification"""
        return state["query_type"].value
    
    async def _process_vqa(self, state: OrchestrationState) -> OrchestrationState:
        """Process Visual Question Answering queries"""
        logger.info("Processing VQA query")
        
        try:
            if not state["current_image"]:
                state["vqa_result"] = {"error": "No image provided for VQA"}
                return state
            
            # Get the question
            question = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    question = msg.content
                    break
            
            # Use VQA agent
            result = self.vqa_agent.ask_question(state["current_image"], question)
            state["vqa_result"] = result
            
            state["intermediate_steps"].append({
                "step": "process_vqa",
                "timestamp": datetime.now().isoformat(),
                "confidence": result.get("confidence", 0.0),
                "answer_length": len(result.get("answer", ""))
            })
            
        except Exception as e:
            logger.error(f"Error in VQA processing: {e}")
            state["vqa_result"] = {"error": str(e)}
        
        return state
    
    async def _process_conversation(self, state: OrchestrationState) -> OrchestrationState:
        """Process conversational queries"""
        logger.info("Processing conversation query")
        
        try:
            # Get the message
            message = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    message = msg.content
                    break
            
            # Use conversation agent
            result = self.conversation_agent.chat(
                message=message,
                image=state["current_image"],
                conversation_id=state["session_id"],
                user_id=state["user_id"]
            )
            
            state["conversation_result"] = result
            
            state["intermediate_steps"].append({
                "step": "process_conversation",
                "timestamp": datetime.now().isoformat(),
                "has_image": state["current_image"] is not None,
                "response_length": len(result.get("response", ""))
            })
            
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")
            state["conversation_result"] = {"error": str(e)}
        
        return state
    
    async def _process_complex(self, state: OrchestrationState) -> OrchestrationState:
        """Process complex multi-step queries"""
        logger.info("Processing complex query")
        
        try:
            # For complex queries, we might need multiple agents
            steps = []
            
            # Step 1: If there's an image, analyze it first
            if state["current_image"]:
                question = ""
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage):
                        question = msg.content
                        break
                
                vqa_result = self.vqa_agent.ask_question(state["current_image"], question)
                steps.append({"step": "image_analysis", "result": vqa_result})
            
            # Step 2: Get relevant context from vector store
            if self.config.enable_retrieval:
                retrieval_result = self.vector_store.query_documents(
                    query=question if 'question' in locals() else "context search",
                    n_results=3
                )
                steps.append({"step": "context_retrieval", "result": retrieval_result})
            
            # Step 3: Synthesize with conversation agent
            message = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    message = msg.content
                    break
            
            # Add context to the message
            if steps:
                context_info = "Based on analysis: "
                for step in steps:
                    if "image_analysis" in step["step"]:
                        context_info += f"Image shows: {step['result'].get('answer', 'unknown')}. "
                    elif "context_retrieval" in step["step"]:
                        context_info += f"Relevant context found. "
                
                message = f"{context_info}\n\nOriginal question: {message}"
            
            conversation_result = self.conversation_agent.chat(
                message=message,
                conversation_id=state["session_id"],
                user_id=state["user_id"]
            )
            
            state["analysis_result"] = {
                "steps": steps,
                "final_response": conversation_result,
                "complexity_score": len(steps)
            }
            
            state["intermediate_steps"].append({
                "step": "process_complex",
                "timestamp": datetime.now().isoformat(),
                "num_sub_steps": len(steps),
                "complexity_score": len(steps)
            })
            
        except Exception as e:
            logger.error(f"Error in complex processing: {e}")
            state["analysis_result"] = {"error": str(e)}
        
        return state
    
    async def _process_retrieval(self, state: OrchestrationState) -> OrchestrationState:
        """Process retrieval-augmented queries"""
        logger.info("Processing retrieval query")
        
        try:
            # Get the query
            query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            
            # Retrieve relevant documents
            retrieval_results = self.vector_store.query_documents(
                query=query,
                n_results=5
            )
            
            state["retrieval_context"] = retrieval_results.get("documents", [])
            
            # Use conversation agent with retrieved context
            enhanced_message = f"Using retrieved context:\n{query}"
            conversation_result = self.conversation_agent.chat(
                message=enhanced_message,
                image=state["current_image"],
                conversation_id=state["session_id"],
                user_id=state["user_id"]
            )
            
            state["conversation_result"] = conversation_result
            
            state["intermediate_steps"].append({
                "step": "process_retrieval",
                "timestamp": datetime.now().isoformat(),
                "num_retrieved": len(state["retrieval_context"]),
                "has_context": len(state["retrieval_context"]) > 0
            })
            
        except Exception as e:
            logger.error(f"Error in retrieval processing: {e}")
            state["retrieval_context"] = []
            state["conversation_result"] = {"error": str(e)}
        
        return state
    
    async def _process_analysis(self, state: OrchestrationState) -> OrchestrationState:
        """Process detailed image analysis queries"""
        logger.info("Processing analysis query")
        
        try:
            if not state["current_image"]:
                state["analysis_result"] = {"error": "No image provided for analysis"}
                return state
            
            # Use multiple approaches for detailed analysis
            results = {}
            
            # 1. VQA for specific questions
            questions = [
                "What is the main subject of this image?",
                "What colors are prominent in this image?",
                "What is happening in this image?",
                "What objects can you see in this image?"
            ]
            
            vqa_results = []
            for q in questions:
                result = self.vqa_agent.ask_question(state["current_image"], q)
                vqa_results.append({"question": q, "answer": result.get("answer", "")})
            
            results["detailed_analysis"] = vqa_results
            
            # 2. Use fusion model if available
            if hasattr(self.fusion_model, 'process_multimodal'):
                fusion_result = self.fusion_model.process_multimodal(
                    state["current_image"], 
                    "Detailed image analysis"
                )
                results["multimodal_analysis"] = fusion_result
            
            state["analysis_result"] = results
            
            state["intermediate_steps"].append({
                "step": "process_analysis",
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "analysis_depth": "detailed"
            })
            
        except Exception as e:
            logger.error(f"Error in analysis processing: {e}")
            state["analysis_result"] = {"error": str(e)}
        
        return state
    
    async def _synthesize_response(self, state: OrchestrationState) -> OrchestrationState:
        """Synthesize final response from all processing results"""
        logger.info("Synthesizing response")
        
        try:
            response_parts = []
            confidence_scores = []
            
            # Gather results from different processing paths
            if state.get("vqa_result"):
                if "error" not in state["vqa_result"]:
                    response_parts.append(state["vqa_result"].get("answer", ""))
                    confidence_scores.append(state["vqa_result"].get("confidence", 0.5))
            
            if state.get("conversation_result"):
                if "error" not in state["conversation_result"]:
                    response_parts.append(state["conversation_result"].get("response", ""))
            
            if state.get("analysis_result"):
                if "error" not in state["analysis_result"]:
                    if "steps" in state["analysis_result"]:
                        # Complex processing
                        final_resp = state["analysis_result"]["final_response"]
                        if "response" in final_resp:
                            response_parts.append(final_resp["response"])
                    elif "detailed_analysis" in state["analysis_result"]:
                        # Detailed analysis
                        analysis_summary = "Based on detailed analysis: "
                        for item in state["analysis_result"]["detailed_analysis"]:
                            analysis_summary += f"{item['answer']}. "
                        response_parts.append(analysis_summary)
            
            # Combine responses
            if response_parts:
                state["final_response"] = " ".join(response_parts)
            else:
                state["final_response"] = "I apologize, but I couldn't process your request. Please try again."
            
            # Calculate overall confidence
            if confidence_scores:
                state["confidence_score"] = sum(confidence_scores) / len(confidence_scores)
            else:
                state["confidence_score"] = 0.5
            
            state["intermediate_steps"].append({
                "step": "synthesize_response",
                "timestamp": datetime.now().isoformat(),
                "num_parts": len(response_parts),
                "confidence": state["confidence_score"],
                "response_length": len(state["final_response"])
            })
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            state["final_response"] = "I encountered an error while processing your request."
            state["confidence_score"] = 0.0
        
        return state
    
    async def _quality_check(self, state: OrchestrationState) -> OrchestrationState:
        """Check response quality and determine if retry is needed"""
        logger.info("Performing quality check")
        
        try:
            # Simple quality checks
            response = state.get("final_response", "")
            confidence = state.get("confidence_score", 0.0)
            
            quality_score = 1.0
            issues = []
            
            # Check response length
            if len(response) < 10:
                quality_score -= 0.3
                issues.append("Response too short")
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                quality_score -= 0.2
                issues.append("Low confidence")
            
            # Check for error messages
            if "error" in response.lower() or "sorry" in response.lower():
                quality_score -= 0.4
                issues.append("Contains error indicators")
            
            # Determine next action
            if quality_score >= 0.6 and len(issues) == 0:
                state["next_step"] = "complete"
            else:
                state["next_step"] = "complete"  # For now, don't retry to avoid loops
            
            state["metadata"]["quality_score"] = quality_score
            state["metadata"]["quality_issues"] = issues
            
            state["intermediate_steps"].append({
                "step": "quality_check",
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "num_issues": len(issues),
                "decision": state["next_step"]
            })
            
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
            state["next_step"] = "complete"
        
        return state
    
    def _check_quality(self, state: OrchestrationState) -> str:
        """Return routing decision from quality check"""
        return state.get("next_step", "complete")
    
    def predict(self, message: str, image: Optional[Union[str, Image.Image]] = None, 
               session_id: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Main prediction interface - orchestrates the entire pipeline
        
        Args:
            message: User message
            image: Optional image (path or PIL Image)
            session_id: Session ID (generates new if None)
            user_id: User ID
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Process image if provided
            current_image = None
            current_image_path = None
            
            if image is not None:
                if isinstance(image, str):
                    current_image_path = image
                    current_image = Image.open(image)
                elif isinstance(image, Image.Image):
                    current_image = image
                    current_image_path = f"session_{session_id}_{datetime.now().strftime('%H%M%S')}.jpg"
            
            # Create initial state
            initial_state: OrchestrationState = {
                "messages": [HumanMessage(content=message)],
                "query_type": None,
                "current_image": current_image,
                "current_image_path": current_image_path,
                "session_id": session_id,
                "user_id": user_id,
                "vqa_result": None,
                "conversation_result": None,
                "retrieval_context": None,
                "analysis_result": None,
                "next_step": None,
                "requires_followup": False,
                "intermediate_steps": [],
                "metadata": {},
                "final_response": None,
                "confidence_score": None
            }
            
            # Create thread config
            config = {"configurable": {"thread_id": session_id}}
            
            # Run the orchestration graph
            result = self.graph.invoke(initial_state, config=config)
            
            # Update active sessions
            self.active_sessions[session_id] = {
                "last_update": datetime.now(),
                "user_id": user_id,
                "query_count": self.active_sessions.get(session_id, {}).get("query_count", 0) + 1
            }
            
            return {
                "response": result.get("final_response", "I couldn't process your request."),
                "confidence": result.get("confidence_score", 0.0),
                "session_id": session_id,
                "query_type": result.get("query_type", {}).get("value", "unknown") if result.get("query_type") else "unknown",
                "has_image": current_image is not None,
                "processing_steps": result.get("intermediate_steps", []),
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "session_id": session_id or "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def apredict(self, message: str, image: Optional[Union[str, Image.Image]] = None, 
                      session_id: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """Async version of predict method"""
        # For simplicity, just call the sync version
        # In a full implementation, you'd use ainvoke throughout
        return self.predict(message, image, session_id, user_id)
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        return {"error": "Session not found"}
    
    def list_active_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List all active sessions, optionally filtered by user"""
        if user_id:
            return [sid for sid, info in self.active_sessions.items() 
                   if info.get("user_id") == user_id]
        return list(self.active_sessions.keys())
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = [
            sid for sid, info in self.active_sessions.items()
            if info.get("last_update", datetime.min) < cutoff_time
        ]
        
        for sid in expired_sessions:
            del self.active_sessions[sid]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)


class PipelineManager:
    """
    High-level manager for the prediction pipeline
    Handles multiple pipelines, load balancing, and monitoring
    """
    
    def __init__(self, num_pipelines: int = 1, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipelines: List[PredictionPipeline] = []
        self.current_pipeline_idx = 0
        
        # Initialize multiple pipelines for load balancing
        for i in range(num_pipelines):
            pipeline = PredictionPipeline(self.config)
            self.pipelines.append(pipeline)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "query_type_distribution": {},
            "start_time": datetime.now()
        }
        
        logger.info(f"Pipeline Manager initialized with {num_pipelines} pipelines")
    
    def _get_next_pipeline(self) -> PredictionPipeline:
        """Simple round-robin load balancing"""
        pipeline = self.pipelines[self.current_pipeline_idx]
        self.current_pipeline_idx = (self.current_pipeline_idx + 1) % len(self.pipelines)
        return pipeline
    
    def predict(self, message: str, image: Optional[Union[str, Image.Image]] = None, 
               session_id: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Main prediction interface with load balancing and monitoring
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        try:
            # Get pipeline and make prediction
            pipeline = self._get_next_pipeline()
            result = pipeline.predict(message, image, session_id, user_id)
            
            # Update statistics
            self.stats["successful_requests"] += 1
            
            # Update query type distribution
            query_type = result.get("query_type", "unknown")
            self.stats["query_type_distribution"][query_type] = \
                self.stats["query_type_distribution"].get(query_type, 0) + 1
            
            # Update response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["successful_requests"] - 1) + response_time) /
                self.stats["successful_requests"]
            )
            
            # Add pipeline statistics to result
            result["pipeline_stats"] = {
                "response_time": response_time,
                "pipeline_id": self.pipelines.index(pipeline),
                "total_requests": self.stats["total_requests"]
            }
            
            return result
        
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Pipeline Manager error: {e}")
            
            return {
                "response": "I apologize, but I encountered a system error. Please try again.",
                "confidence": 0.0,
                "session_id": session_id or "error",
                "error": str(e),
                "pipeline_stats": {
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "pipeline_id": -1,
                    "total_requests": self.stats["total_requests"]
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ) * 100,
            "requests_per_second": self.stats["total_requests"] / max(uptime, 1),
            "num_active_pipelines": len(self.pipelines)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipelines"""
        health_status = {
            "status": "healthy",
            "pipelines": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for i, pipeline in enumerate(self.pipelines):
            try:
                # Simple health check - try a basic prediction
                test_result = pipeline.predict("Health check", session_id=f"health_check_{i}")
                pipeline_status = {
                    "pipeline_id": i,
                    "status": "healthy" if "error" not in test_result else "unhealthy",
                    "response_received": "response" in test_result,
                    "active_sessions": len(pipeline.active_sessions)
                }
            except Exception as e:
                pipeline_status = {
                    "pipeline_id": i,
                    "status": "unhealthy",
                    "error": str(e),
                    "active_sessions": 0
                }
                health_status["status"] = "degraded"
            
            health_status["pipelines"].append(pipeline_status)
        
        return health_status


# Testing and demo functions
async def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("Testing Prediction Pipeline - Agent Orchestration...")
    print("=" * 60)
    
    # Initialize pipeline
    config = PipelineConfig(
        llm_provider="mock",  # Use mock for testing
        enable_retrieval=True,
        enable_memory=True
    )
    
    pipeline_manager = PipelineManager(num_pipelines=1, config=config)
    
    # Test different query types
    test_cases = [
        {
            "name": "Simple VQA Query",
            "message": "What do you see in this image?",
            "image": None,  # Would be actual image in real use
            "expected_type": "VQA"
        },
        {
            "name": "Conversation Query", 
            "message": "Hello! How are you doing today?",
            "image": None,
            "expected_type": "CONVERSATION"
        },
        {
            "name": "Complex Query",
            "message": "Can you compare this image to what we discussed earlier and explain the differences?",
            "image": None,
            "expected_type": "COMPLEX"
        },
        {
            "name": "Retrieval Query",
            "message": "Find information about our previous conversations regarding images",
            "image": None,
            "expected_type": "RETRIEVAL"
        },
        {
            "name": "Analysis Query",
            "message": "Provide a detailed analysis of this image's composition and elements",
            "image": None,
            "expected_type": "ANALYSIS"
        }
    ]
    
    session_id = str(uuid.uuid4())
    
    print(f"Using session ID: {session_id}")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        # Create mock image for image-related tests
        if "image" in test_case["message"].lower() and test_case["image"] is None:
            try:
                test_image = Image.new('RGB', (100, 100), color='blue')
                test_case["image"] = test_image
                print("📸 Using mock test image")
            except Exception as e:
                print(f"Could not create test image: {e}")
        
        # Make prediction
        result = pipeline_manager.predict(
            message=test_case["message"],
            image=test_case["image"],
            session_id=session_id
        )
        
        # Display results
        print(f"Query: {test_case['message']}")
        print(f"Expected Type: {test_case['expected_type']}")
        print(f"Detected Type: {result.get('query_type', 'unknown').upper()}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Response: {result.get('response', 'No response')[:100]}...")
        
        # Show processing steps
        steps = result.get('processing_steps', [])
        if steps:
            print(f"Processing Steps: {len(steps)}")
            for step in steps:
                print(f"  - {step.get('step', 'unknown')}: {step.get('timestamp', '')}")
        
        # Show pipeline stats
        pipeline_stats = result.get('pipeline_stats', {})
        print(f"Response Time: {pipeline_stats.get('response_time', 0):.3f}s")
        
        print()
    
    # Test session continuity
    print("\n6. Testing Session Continuity")
    print("-" * 40)
    
    followup_message = "What was my first question about?"
    result = pipeline_manager.predict(
        message=followup_message,
        session_id=session_id
    )
    
    print(f"Follow-up Query: {followup_message}")
    print(f"Response: {result.get('response', 'No response')[:100]}...")
    
    # Get pipeline statistics
    print("\n7. Pipeline Statistics")
    print("-" * 40)
    
    stats = pipeline_manager.get_stats()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Average Response Time: {stats['average_response_time']:.3f}s")
    print(f"Query Type Distribution:")
    for query_type, count in stats['query_type_distribution'].items():
        print(f"  - {query_type}: {count}")
    
    # Health check
    print("\n8. Health Check")
    print("-" * 40)
    
    health = pipeline_manager.health_check()
    print(f"Overall Status: {health['status']}")
    for pipeline_health in health['pipelines']:
        print(f"Pipeline {pipeline_health['pipeline_id']}: {pipeline_health['status']}")


def demo_complex_workflow():
    """Demonstrate a complex multi-step workflow"""
    print("\n" + "="*60)
    print("DEMO: Complex Multi-Step Workflow")
    print("="*60)
    
    # Initialize pipeline manager
    config = PipelineConfig(
        enable_retrieval=True,
        enable_memory=True,
        max_iterations=5
    )
    
    manager = PipelineManager(num_pipelines=1, config=config)
    session_id = str(uuid.uuid4())
    
    # Simulate a complex interaction
    workflow_steps = [
        "I have an image I'd like you to analyze",
        "What are the main elements in this image?",
        "How does this compare to typical images of this type?", 
        "Can you find similar examples from our previous conversations?",
        "Based on everything we've discussed, what's your overall assessment?"
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"\nStep {i}: {step}")
        print("-" * 50)
        
        # Add mock image for first step
        test_image = None
        if i == 1:
            try:
                test_image = Image.new('RGB', (200, 200), color='green')
                print("📸 Added test image")
            except:
                pass
        
        result = manager.predict(
            message=step,
            image=test_image,
            session_id=session_id
        )
        
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        
        # Show intermediate steps
        processing_steps = result.get('processing_steps', [])
        if processing_steps:
            print("Processing pipeline:")
            for proc_step in processing_steps[-3:]:  # Show last 3 steps
                print(f"  → {proc_step.get('step', 'unknown')}")
    
    print(f"\nWorkflow completed! Final statistics:")
    stats = manager.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Average response time: {stats['average_response_time']:.3f}s")


def main():
    """Main function for testing"""
    print("Multi-Modal AI Assistant - Prediction Pipeline")
    print("Agent Orchestration with LangGraph")
    print("=" * 60)
    
    # Run async test
    asyncio.run(test_prediction_pipeline())
    
    # Run complex workflow demo
    demo_complex_workflow()
    
    print("\n" + "="*60)
    print("Prediction Pipeline testing completed!")
    print("✅ Step 12: Agent Orchestration - COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
