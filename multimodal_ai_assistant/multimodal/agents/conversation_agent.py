"""conversation_agent.py module"""

"""
Multi-Modal AI Assistant - Conversation Agent
LangGraph-based agent for multi-turn conversations with memory
"""

import os
import sys
import json
import uuid
import asyncio
from typing import List, Dict, Union, Optional, Any, TypedDict, Annotated
import logging
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

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
    from langchain_community.memory import ConversationBufferWindowMemory
    from langchain_core.memory import BaseMemory
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
            
        def set_entry_point(self, node): 
            return self
            
        def compile(self, checkpointer=None): 
            self.checkpointer = checkpointer
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph): 
            self.graph = graph
            
        async def ainvoke(self, input_data, config=None):
            return {"messages": [{"content": "Mock conversation response"}]}
            
        def invoke(self, input_data, config=None):
            return {"messages": [{"content": "Mock conversation response"}]}
    
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
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from multimodal.fusion.multimodal_fusion import MultiModalFusion, FusionConfig
    from vector_store.chroma_store import ChromaStore, ChromaConfig, MultiModalDocument
    from multimodal.agents.vqa_agent import VQAAgent, VQAConfig
except ImportError:
    print("Warning: Custom components not found. Using mock classes.")
    
    class MultiModalFusion:
        def __init__(self, config=None): pass
        def encode_multimodal(self, image, text): 
            return np.random.rand(256)
        def visual_question_answering(self, image, question):
            return {"answer": "Mock answer", "confidence": 0.8}
    
    class ChromaStore:
        def __init__(self, config=None): pass
        def add_document(self, doc): return "doc_id"
        def query_documents(self, query, n_results=5): 
            return {"documents": []}
    
    class VQAAgent:
        def __init__(self, config=None): pass
        def ask_question(self, image, question):
            return {"answer": "Mock VQA answer", "confidence": 0.8}
    
    class VQAConfig:
        def __init__(self): pass
    
    class FusionConfig:
        def __init__(self): pass
    
    class ChromaConfig:
        def __init__(self): pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for conversation
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_image: Optional[Image.Image]
    current_image_path: Optional[str]
    conversation_id: str
    user_id: str
    context_summary: Optional[str]
    relevant_history: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class MockLLM:
    def invoke(self, messages):
        return AIMessage(content="This is a mock response from the conversation agent.")

@dataclass
class ConversationConfig:
    """Configuration for Conversation Agent"""
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    openai_api_key: Optional[str] = None
    
    # Memory settings
    memory_window_size: int = 10
    max_conversation_length: int = 100
    summarize_after: int = 20
    
    # Retrieval settings
    enable_retrieval: bool = True
    max_retrieved_memories: int = 5
    memory_similarity_threshold: float = 0.7
    
    # Component configs
    vqa_config: Optional[VQAConfig] = None
    fusion_config: Optional[FusionConfig] = None
    chroma_config: Optional[ChromaConfig] = None
    
    # System prompt
    system_prompt: str = """You are a helpful AI assistant capable of understanding and discussing images. 
    You maintain context across conversations and can refer to previous images and discussions.
    Be conversational, helpful, and remember what we've talked about before.
    If you can't see an image clearly, ask for clarification or more details."""

@dataclass
class ConversationMemory:
    """Represents a conversation memory entry"""
    id: str
    timestamp: datetime
    user_message: str
    ai_response: str
    image_path: Optional[str] = None
    image_description: Optional[str] = None
    conversation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationAgent:
    """
    LangGraph-based Conversation Agent with multimodal memory
    Maintains context across turns and can discuss images conversationally
    """
    
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        
        # Initialize components
        self.llm = None
        self.vqa_agent = None
        self.fusion_model = None
        self.memory_store = None
        self.graph = None
        self.checkpointer = None
        
        # Active conversations and memories
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_memories: Dict[str, List[ConversationMemory]] = {}
        
        self._initialize_components()
        self._build_graph()
        
        logger.info("Conversation Agent initialized successfully")
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize LLM
            if self.config.llm_provider == "openai":
                if not self.config.openai_api_key:
                    self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
                
                if self.config.openai_api_key:
                    self.llm = ChatOpenAI(
                        api_key=self.config.openai_api_key,
                        model_name=self.config.model_name,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                else:
                    logger.warning("OpenAI API key not found, using mock LLM")
                    self.llm = MockLLM()
            else:
                # Use Ollama
                self.llm = Ollama(model=self.config.model_name, temperature=self.config.temperature)
            
            # Initialize VQA Agent
            if self.config.vqa_config:
                self.vqa_agent = VQAAgent(self.config.vqa_config)
            else:
                self.vqa_agent = VQAAgent()
            
            # Initialize Fusion Model
            if self.config.fusion_config:
                self.fusion_model = MultiModalFusion(self.config.fusion_config)
            else:
                self.fusion_model = MultiModalFusion()
            
            # Initialize Memory Store
            if self.config.chroma_config:
                self.memory_store = ChromaStore(self.config.chroma_config)
            else:
                self.memory_store = ChromaStore()
            
        except Exception as e:
            logger.warning(f"Error initializing components: {e}. Using mock components.")
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components for testing"""
        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="This is a mock response from the conversation agent.")
        
        self.llm = MockLLM()
        self.vqa_agent = VQAAgent()
        self.fusion_model = MultiModalFusion()
        self.memory_store = ChromaStore()
    
    def _build_graph(self):
        """Build the LangGraph conversation workflow"""
        # Initialize checkpointer for memory
        self.checkpointer = MemorySaver()
        
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("analyze_image", self._analyze_image)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_memory", self._save_memory)
        
        # Define edges
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_image")
        workflow.add_edge("analyze_image", "generate_response")
        workflow.add_edge("generate_response", "save_memory")
        workflow.add_edge("save_memory", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    async def _process_input(self, state: ConversationState) -> ConversationState:
        """Process the input message and update state"""
        logger.info("Processing input message")
        
        # Initialize conversation if needed
        if state["conversation_id"] not in self.active_conversations:
            self.active_conversations[state["conversation_id"]] = {
                "start_time": datetime.now(),
                "turn_count": 0,
                "last_image": None
            }
        
        # Update turn count
        self.active_conversations[state["conversation_id"]]["turn_count"] += 1
        
        # Add intermediate step
        state["intermediate_steps"].append({
            "step": "process_input",
            "timestamp": datetime.now().isoformat(),
            "details": "Input processed successfully"
        })
        
        return state
    
    async def _retrieve_context(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant conversation context"""
        logger.info("Retrieving conversation context")
        
        if not self.config.enable_retrieval:
            state["relevant_history"] = []
            return state
        
        try:
            # Get recent conversation history
            conversation_id = state["conversation_id"]
            if conversation_id in self.conversation_memories:
                recent_memories = self.conversation_memories[conversation_id][-self.config.max_retrieved_memories:]
                state["relevant_history"] = [
                    {
                        "user_message": mem.user_message,
                        "ai_response": mem.ai_response,
                        "timestamp": mem.timestamp.isoformat(),
                        "has_image": mem.image_path is not None
                    }
                    for mem in recent_memories
                ]
            else:
                state["relevant_history"] = []
            
            # Add context summary if conversation is long
            if len(state["messages"]) > self.config.summarize_after:
                state["context_summary"] = self._create_context_summary(state)
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["relevant_history"] = []
        
        state["intermediate_steps"].append({
            "step": "retrieve_context",
            "timestamp": datetime.now().isoformat(),
            "retrieved_memories": len(state["relevant_history"])
        })
        
        return state
    
    async def _analyze_image(self, state: ConversationState) -> ConversationState:
        """Analyze current image if present"""
        logger.info("Analyzing current image")
        
        if state["current_image"] is None:
            state["intermediate_steps"].append({
                "step": "analyze_image",
                "timestamp": datetime.now().isoformat(),
                "details": "No image to analyze"
            })
            return state
        
        try:
            # Get the latest user message
            last_message = state["messages"][-1] if state["messages"] else None
            if last_message and hasattr(last_message, 'content'):
                question = last_message.content
            else:
                question = "What do you see in this image?"
            
            # Use VQA agent to analyze the image
            vqa_result = self.vqa_agent.ask_question(state["current_image"], question)
            
            # Store image analysis in metadata
            state["metadata"]["image_analysis"] = {
                "question": question,
                "answer": vqa_result.get("answer", "Unable to analyze image"),
                "confidence": vqa_result.get("confidence", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update active conversation with current image
            self.active_conversations[state["conversation_id"]]["last_image"] = state["current_image_path"]
        
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            state["metadata"]["image_analysis"] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        state["intermediate_steps"].append({
            "step": "analyze_image",
            "timestamp": datetime.now().isoformat(),
            "details": "Image analysis completed"
        })
        
        return state
    
    async def _generate_response(self, state: ConversationState) -> ConversationState:
        """Generate conversational response"""
        logger.info("Generating response")
        
        try:
            # Build context for the LLM
            context_parts = [self.config.system_prompt]
            
            # Add context summary if available
            if state["context_summary"]:
                context_parts.append(f"Previous conversation summary: {state['context_summary']}")
            
            # Add recent history
            if state["relevant_history"]:
                context_parts.append("Recent conversation history:")
                for hist in state["relevant_history"][-3:]:  # Last 3 exchanges
                    context_parts.append(f"User: {hist['user_message']}")
                    context_parts.append(f"Assistant: {hist['ai_response']}")
            
            # Add image analysis if available
            if "image_analysis" in state["metadata"]:
                analysis = state["metadata"]["image_analysis"]
                if "answer" in analysis:
                    context_parts.append(f"Current image analysis: {analysis['answer']}")
            
            # Build messages for LLM
            messages = [SystemMessage(content="\n\n".join(context_parts))]
            
            # Add recent conversation messages
            recent_messages = state["messages"][-self.config.memory_window_size:]
            messages.extend(recent_messages)
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Add AI response to messages
            if hasattr(response, 'content'):
                ai_message = AIMessage(content=response.content)
            else:
                ai_message = AIMessage(content=str(response))
            
            state["messages"].append(ai_message)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_message = AIMessage(content="I apologize, but I encountered an error generating a response. Could you please try again?")
            state["messages"].append(error_message)
        
        state["intermediate_steps"].append({
            "step": "generate_response",
            "timestamp": datetime.now().isoformat(),
            "details": "Response generated successfully"
        })
        
        return state
    
    async def _save_memory(self, state: ConversationState) -> ConversationState:
        """Save conversation turn to memory"""
        logger.info("Saving conversation memory")
        
        try:
            # Get the last user and AI messages
            user_message = ""
            ai_message = ""
            
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and not ai_message:
                    ai_message = msg.content
                elif isinstance(msg, HumanMessage) and not user_message:
                    user_message = msg.content
                    break
            
            # Create memory entry
            memory = ConversationMemory(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_message=user_message,
                ai_response=ai_message,
                image_path=state["current_image_path"],
                conversation_id=state["conversation_id"],
                metadata={
                    "turn_count": self.active_conversations[state["conversation_id"]]["turn_count"],
                    "has_image": state["current_image"] is not None,
                    "image_analysis": state["metadata"].get("image_analysis")
                }
            )
            
            # Store in conversation memories
            if state["conversation_id"] not in self.conversation_memories:
                self.conversation_memories[state["conversation_id"]] = []
            
            self.conversation_memories[state["conversation_id"]].append(memory)
            
            # Optionally save to vector store for retrieval
            if self.memory_store and user_message and ai_message:
                doc_content = f"User: {user_message}\nAssistant: {ai_message}"
                # Note: In a full implementation, you'd create a proper MultiModalDocument
                # self.memory_store.add_document(MultiModalDocument(...))
        
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
        
        state["intermediate_steps"].append({
            "step": "save_memory",
            "timestamp": datetime.now().isoformat(),
            "details": "Memory saved successfully"
        })
        
        return state
    
    def _create_context_summary(self, state: ConversationState) -> str:
        """Create a summary of the conversation context"""
        try:
            conversation_id = state["conversation_id"]
            if conversation_id not in self.conversation_memories:
                return ""
            
            memories = self.conversation_memories[conversation_id]
            if len(memories) < 3:
                return ""
            
            # Simple summary - in practice, you might use the LLM to create better summaries
            recent_topics = []
            for memory in memories[-10:]:  # Last 10 exchanges
                if memory.image_path:
                    recent_topics.append(f"discussed image: {memory.image_path}")
                if len(memory.user_message) > 20:
                    recent_topics.append(f"user asked about: {memory.user_message[:50]}...")
            
            return f"Recent conversation included: {'; '.join(recent_topics[-5:])}"
        
        except Exception as e:
            logger.error(f"Error creating context summary: {e}")
            return ""
    
    def chat(self, message: str, image: Optional[Union[str, Image.Image]] = None, 
             conversation_id: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Main chat interface for single turn conversation
        
        Args:
            message: User message
            image: Optional image (path or PIL Image)
            conversation_id: Conversation ID (generates new if None)
            user_id: User ID
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Generate conversation ID if not provided
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            
            # Process image if provided
            current_image = None
            current_image_path = None
            
            if image is not None:
                if isinstance(image, str):
                    current_image_path = image
                    current_image = Image.open(image)
                elif isinstance(image, Image.Image):
                    current_image = image
                    current_image_path = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Create initial state
            initial_state: ConversationState = {
                "messages": [HumanMessage(content=message)],
                "current_image": current_image,
                "current_image_path": current_image_path,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "context_summary": None,
                "relevant_history": [],
                "intermediate_steps": [],
                "metadata": {}
            }
            
            # Create thread config
            config = {"configurable": {"thread_id": conversation_id}}
            
            # Run the graph
            result = self.graph.invoke(initial_state, config=config)
            
            # Extract response
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            response = ai_messages[-1].content if ai_messages else "I apologize, but I couldn't generate a response."
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "has_image": current_image is not None,
                "image_analysis": result["metadata"].get("image_analysis"),
                "turn_count": self.active_conversations.get(conversation_id, {}).get("turn_count", 0),
                "intermediate_steps": result["intermediate_steps"]
            }
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "conversation_id": conversation_id or "error",
                "error": str(e)
            }
    
    async def achat(self, message: str, image: Optional[Union[str, Image.Image]] = None, 
                   conversation_id: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """Async version of chat method"""
        # For simplicity, just call the sync version
        # In a full implementation, you'd use ainvoke throughout
        return self.chat(message, image, conversation_id, user_id)
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        if conversation_id not in self.conversation_memories:
            return []
        
        return [
            {
                "id": mem.id,
                "timestamp": mem.timestamp.isoformat(),
                "user_message": mem.user_message,
                "ai_response": mem.ai_response,
                "has_image": mem.image_path is not None,
                "image_path": mem.image_path
            }
            for mem in self.conversation_memories[conversation_id]
        ]
    
    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation"""
        if conversation_id in self.conversation_memories:
            del self.conversation_memories[conversation_id]
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
    
    def list_conversations(self, user_id: str) -> List[str]:
        """List all conversation IDs for a user"""
        # In a full implementation, you'd filter by user_id
        return list(self.conversation_memories.keys())


# Testing and demo functions
async def test_conversation_agent():
    """Test the conversation agent"""
    print("Testing Conversation Agent...")
    
    # Initialize agent
    config = ConversationConfig(
        llm_provider="mock",  # Use mock for testing
        temperature=0.7
    )
    agent = ConversationAgent(config)
    
    # Test basic conversation
    print("\n1. Testing basic conversation:")
    response1 = agent.chat("Hello, how are you today?")
    print(f"Response: {response1['response']}")
    conversation_id = response1['conversation_id']
    
    # Continue conversation
    print("\n2. Continuing conversation:")
    response2 = agent.chat(
        "What's the weather like?", 
        conversation_id=conversation_id
    )
    print(f"Response: {response2['response']}")
    
    # Test with image
    print("\n3. Testing with mock image:")
    try:
        # Create a small test image
        test_image = Image.new('RGB', (100, 100), color='red')
        response3 = agent.chat(
            "What do you see in this image?", 
            image=test_image,
            conversation_id=conversation_id
        )
        print(f"Response: {response3['response']}")
        print(f"Image analysis: {response3.get('image_analysis', 'None')}")
    except Exception as e:
        print(f"Image test failed: {e}")
    
    # Get conversation history
    print("\n4. Conversation history:")
    history = agent.get_conversation_history(conversation_id)
    print(f"Total turns: {len(history)}")
    for i, turn in enumerate(history):
        print(f"Turn {i+1}: User: {turn['user_message'][:50]}...")
        print(f"         AI: {turn['ai_response'][:50]}...")

def main():
    """Main function for testing"""
    print("Multi-Modal Conversation Agent")
    print("=" * 50)
    
    # Run async test
    asyncio.run(test_conversation_agent())
    
    print("\nConversation Agent test completed!")

if __name__ == "__main__":
    main()
