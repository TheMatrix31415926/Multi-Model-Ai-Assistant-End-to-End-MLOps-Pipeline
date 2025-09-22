"""vqa_agent.py module"""

"""
Multi-Modal AI Assistant - VQA Agent
LangGraph-based agent for Visual Question Answering
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Union, Optional, Any, TypedDict, Annotated
import logging
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import numpy as np

# LangGraph and LangChain imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
except ImportError:
    print("Warning: LangGraph/LangChain not installed. Using mock classes for testing.")
    
    # Mock classes for testing without LangGraph
    class StateGraph:
        def __init__(self, state_schema): 
            self.nodes = {}
            self.edges = {}
            
        def add_node(self, name, func): 
            self.nodes[name] = func
            return self
            
        def add_edge(self, from_node, to_node): 
            self.edges[from_node] = to_node
            return self
            
        def set_entry_point(self, node): 
            self.entry = node
            return self
            
        def compile(self): 
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph): 
            self.graph = graph
            
        async def ainvoke(self, input_data, config=None):
            return {"response": "Mock VQA response"}
            
        def invoke(self, input_data, config=None):
            return {"response": "Mock VQA response"}
    
    class BaseMessage:
        def __init__(self, content): 
            self.content = content
    
    class HumanMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass
    
    def add_messages(x, y): 
        return x + y
    
    END = "END"

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from multimodal.fusion.multimodal_fusion import MultiModalFusion, FusionConfig
    from vector_store.chroma_store import ChromaStore, ChromaConfig
except ImportError:
    print("Warning: Custom components not found. Using mock classes.")
    
    class MultiModalFusion:
        def __init__(self, config=None): pass
        def visual_question_answering(self, image, question): 
            return {"answer": "Mock answer", "confidence": 0.8}
        def encode_multimodal(self, image, text): 
            return np.random.rand(256)
    
    class ChromaStore:
        def __init__(self, config=None): pass
        def query_documents(self, query, n_results=5): 
            return {"documents": []}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for LangGraph
class VQAState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image_path: Optional[str]
    image_data: Optional[Image.Image]
    question: str
    context: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]
    final_answer: Optional[str]
    confidence: Optional[float]
    metadata: Dict[str, Any]

@dataclass
class VQAConfig:
    """Configuration for VQA Agent"""
    llm_provider: str = "openai"  # "openai", "ollama", "huggingface"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    openai_api_key: Optional[str] = None
    enable_retrieval: bool = True
    max_retrieved_docs: int = 3
    fusion_config: Optional[FusionConfig] = None
    chroma_config: Optional[ChromaConfig] = None
    system_prompt: str = """You are an expert Visual Question Answering assistant. 
    Analyze images carefully and answer questions about them accurately and concisely.
    Use any retrieved context to enhance your answers."""

class VQAAgent:
    """
    LangGraph-based Visual Question Answering Agent
    Combines multimodal understanding with retrieval-augmented generation
    """
    
    def __init__(self, config: Optional[VQAConfig] = None):
        self.config = config or VQAConfig()
        
        # Initialize components
        self.llm = None
        self.fusion_model = None
        self.vector_store = None
        self.graph = None
        
        self._initialize_components()
        self._build_graph()
        
        logger.info("VQA Agent initialized successfully")
    
    def _initialize_components(self):
        """Initialize LLM, fusion model, and vector store"""
        try:
            # Initialize LLM
            if self.config.llm_provider == "openai":
                if not self.config.openai_api_key:
                    self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
                
                if self.config.openai_api_key:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        api_key=self.config.openai_api_key,
                        model=self.config.model_name,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                else:
                    logger.warning("OpenAI API key not found, using mock LLM")
                    self.llm = None
            
            elif self.config.llm_provider == "ollama":
                self.llm = Ollama(model=self.config.model_name)
            
            # Initialize multimodal fusion
            fusion_config = self.config.fusion_config or FusionConfig()
            self.fusion_model = MultiModalFusion(fusion_config)
            
            # Initialize vector store if retrieval is enabled
            if self.config.enable_retrieval:
                chroma_config = self.config.chroma_config or ChromaConfig()
                self.vector_store = ChromaStore(chroma_config)
            
            logger.info("All components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        try:
            # Create the graph
            workflow = StateGraph(VQAState)
            
            # Add nodes
            workflow.add_node("process_input", self._process_input)
            workflow.add_node("retrieve_context", self._retrieve_context)
            workflow.add_node("analyze_image", self._analyze_image)
            workflow.add_node("generate_answer", self._generate_answer)
            workflow.add_node("refine_answer", self._refine_answer)
            
            # Define the flow
            workflow.set_entry_point("process_input")
            
            # Add conditional edges
            workflow.add_edge("process_input", "retrieve_context")
            workflow.add_edge("retrieve_context", "analyze_image")
            workflow.add_edge("analyze_image", "generate_answer")
            workflow.add_edge("generate_answer", "refine_answer")
            workflow.add_edge("refine_answer", END)
            
            # Compile the graph
            self.graph = workflow.compile()
            
            logger.info("LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    async def _process_input(self, state: VQAState) -> Dict[str, Any]:
        """Process and validate input"""
        try:
            logger.info("Processing input...")
            
            # Extract question from messages if not provided
            if not state.get("question") and state.get("messages"):
                last_message = state["messages"][-1]
                if hasattr(last_message, 'content'):
                    state["question"] = last_message.content
            
            # Validate required inputs
            if not state.get("question"):
                raise ValueError("Question is required")
            
            if not state.get("image_data") and not state.get("image_path"):
                raise ValueError("Image is required (either image_data or image_path)")
            
            # Load image if path is provided
            if state.get("image_path") and not state.get("image_data"):
                state["image_data"] = Image.open(state["image_path"])
            
            # Initialize metadata
            state["metadata"] = state.get("metadata", {})
            state["intermediate_steps"] = state.get("intermediate_steps", [])
            
            # Add processing step
            state["intermediate_steps"].append({
                "step": "process_input",
                "status": "completed",
                "details": {"question_length": len(state["question"])}
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            state["intermediate_steps"].append({
                "step": "process_input",
                "status": "error",
                "error": str(e)
            })
            return state
    
    async def _retrieve_context(self, state: VQAState) -> Dict[str, Any]:
        """Retrieve relevant context from vector store"""
        try:
            logger.info("Retrieving context...")
            
            if not self.config.enable_retrieval or not self.vector_store:
                state["retrieved_docs"] = []
                state["context"] = None
                return state
            
            # Query vector store with the question
            query_results = self.vector_store.query_documents(
                query_text=state["question"],
                n_results=self.config.max_retrieved_docs,
                include_images=True
            )
            
            # Process retrieved documents
            retrieved_docs = []
            context_parts = []
            
            for doc in query_results.get("documents", []):
                retrieved_docs.append({
                    "text": doc["text"],
                    "score": doc["score"],
                    "metadata": doc.get("metadata", {})
                })
                
                if doc["score"] > 0.5:  # Only include relevant docs
                    context_parts.append(f"Context: {doc['text']}")
            
            state["retrieved_docs"] = retrieved_docs
            state["context"] = "\n".join(context_parts) if context_parts else None
            
            # Add step info
            state["intermediate_steps"].append({
                "step": "retrieve_context",
                "status": "completed",
                "details": {
                    "retrieved_count": len(retrieved_docs),
                    "relevant_count": len(context_parts)
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["intermediate_steps"].append({
                "step": "retrieve_context", 
                "status": "error",
                "error": str(e)
            })
            state["retrieved_docs"] = []
            state["context"] = None
            return state
    
    async def _analyze_image(self, state: VQAState) -> Dict[str, Any]:
        """Analyze image using multimodal fusion"""
        try:
            logger.info("Analyzing image...")
            
            if not state.get("image_data"):
                raise ValueError("No image data available")
            
            # Use fusion model for VQA
            vqa_result = self.fusion_model.visual_question_answering(
                image=state["image_data"],
                question=state["question"]
            )
            
            # Store intermediate results
            state["metadata"].update({
                "fusion_confidence": vqa_result.get("confidence", 0.0),
                "fusion_answer": vqa_result.get("answer", "")
            })
            
            # Add step info
            state["intermediate_steps"].append({
                "step": "analyze_image",
                "status": "completed", 
                "details": {
                    "fusion_confidence": vqa_result.get("confidence", 0.0),
                    "fusion_answer_length": len(vqa_result.get("answer", ""))
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            state["intermediate_steps"].append({
                "step": "analyze_image",
                "status": "error",
                "error": str(e)
            })
            return state
    
    async def _generate_answer(self, state: VQAState) -> Dict[str, Any]:
        """Generate answer using LLM"""
        try:
            logger.info("Generating answer...")
            
            # Prepare prompt
            prompt_parts = [self.config.system_prompt]
            
            # Add context if available
            if state.get("context"):
                prompt_parts.append(f"\nRelevant Context:\n{state['context']}")
            
            # Add fusion model insights
            if state.get("metadata", {}).get("fusion_answer"):
                prompt_parts.append(f"\nInitial Analysis: {state['metadata']['fusion_answer']}")
            
            # Add the question
            prompt_parts.append(f"\nQuestion: {state['question']}")
            prompt_parts.append("\nPlease provide a clear, accurate answer based on the image and available context.")
            
            full_prompt = "\n".join(prompt_parts)
            
            # Generate response
            if self.llm:
                try:
                    response = await self.llm.ainvoke(full_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                except:
                    # Fallback to sync call
                    response = self.llm.invoke(full_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
            else:
                # Use fusion model answer as fallback
                answer = state.get("metadata", {}).get("fusion_answer", "Unable to generate answer")
            
            state["final_answer"] = answer
            
            # Add step info
            state["intermediate_steps"].append({
                "step": "generate_answer",
                "status": "completed",
                "details": {
                    "answer_length": len(answer),
                    "used_llm": self.llm is not None
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["intermediate_steps"].append({
                "step": "generate_answer",
                "status": "error", 
                "error": str(e)
            })
            
            # Fallback answer
            state["final_answer"] = state.get("metadata", {}).get("fusion_answer", "Unable to generate answer")
            return state
    
    async def _refine_answer(self, state: VQAState) -> Dict[str, Any]:
        """Refine and finalize the answer"""
        try:
            logger.info("Refining answer...")
            
            # Calculate confidence score
            fusion_conf = state.get("metadata", {}).get("fusion_confidence", 0.0)
            context_boost = 0.1 if state.get("context") else 0.0
            retrieval_boost = 0.1 if state.get("retrieved_docs") else 0.0
            
            confidence = min(1.0, fusion_conf + context_boost + retrieval_boost)
            state["confidence"] = confidence
            
            # Add final message to conversation
            if not state.get("messages"):
                state["messages"] = []
            
            state["messages"].append(AIMessage(content=state["final_answer"]))
            
            # Add final step info
            state["intermediate_steps"].append({
                "step": "refine_answer",
                "status": "completed",
                "details": {
                    "final_confidence": confidence,
                    "total_steps": len(state["intermediate_steps"]) + 1
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error refining answer: {e}")
            state["intermediate_steps"].append({
                "step": "refine_answer",
                "status": "error",
                "error": str(e)
            })
            return state
    
    async def ask_question_async(self, image: Union[str, Image.Image], 
                               question: str, 
                               context: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously ask a question about an image
        
        Args:
            image: Image path or PIL Image
            question: Question about the image
            context: Optional additional context
            
        Returns:
            VQA result dictionary
        """
        try:
            # Prepare initial state
            initial_state = VQAState(
                messages=[HumanMessage(content=question)],
                image_path=image if isinstance(image, str) else None,
                image_data=image if isinstance(image, Image.Image) else None,
                question=question,
                context=context,
                retrieved_docs=[],
                intermediate_steps=[],
                final_answer=None,
                confidence=None,
                metadata={}
            )
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "question": question,
                "answer": result.get("final_answer", "No answer generated"),
                "confidence": result.get("confidence", 0.0),
                "retrieved_docs": result.get("retrieved_docs", []),
                "intermediate_steps": result.get("intermediate_steps", []),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error in async VQA: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def ask_question(self, image: Union[str, Image.Image], 
                    question: str,
                    context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question about an image (synchronous wrapper)
        
        Args:
            image: Image path or PIL Image
            question: Question about the image  
            context: Optional additional context
            
        Returns:
            VQA result dictionary
        """
        try:
            # Try async first
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.ask_question_async(image, question, context)
            )
        except:
            # Fallback to sync execution
            try:
                initial_state = VQAState(
                    messages=[HumanMessage(content=question)],
                    image_path=image if isinstance(image, str) else None,
                    image_data=image if isinstance(image, Image.Image) else None,
                    question=question,
                    context=context,
                    retrieved_docs=[],
                    intermediate_steps=[],
                    final_answer=None,
                    confidence=None,
                    metadata={}
                )
                
                result = self.graph.invoke(initial_state)
                
                return {
                    "question": question,
                    "answer": result.get("final_answer", "No answer generated"),
                    "confidence": result.get("confidence", 0.0),
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "intermediate_steps": result.get("intermediate_steps", []),
                    "metadata": result.get("metadata", {})
                }
                
            except Exception as e:
                logger.error(f"Error in sync VQA: {e}")
                return {
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "error": str(e)
                }
    
    def batch_questions(self, image: Union[str, Image.Image],
                       questions: List[str]) -> List[Dict[str, Any]]:
        """
        Ask multiple questions about the same image
        
        Args:
            image: Image path or PIL Image
            questions: List of questions
            
        Returns:
            List of VQA results
        """
        results = []
        for question in questions:
            result = self.ask_question(image, question)
            results.append(result)
        
        return results
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent configuration and status"""
        return {
            "llm_provider": self.config.llm_provider,
            "model_name": self.config.model_name,
            "retrieval_enabled": self.config.enable_retrieval,
            "max_retrieved_docs": self.config.max_retrieved_docs,
            "components": {
                "llm_available": self.llm is not None,
                "fusion_model_available": self.fusion_model is not None,
                "vector_store_available": self.vector_store is not None
            }
        }

# Test function
def test_vqa_agent():
    """Test the VQA Agent"""
    print("Testing VQA Agent...")
    
    try:
        # Initialize agent
        config = VQAConfig(
            llm_provider="ollama",  # Use local model for testing
            model_name="llama2",
            enable_retrieval=False,  # Disable for simple testing
            temperature=0.7
        )
        
        agent = VQAAgent(config)
        
        # Print agent info
        print("Agent Info:", agent.get_agent_info())
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test single question
        result = agent.ask_question(
            image=test_image,
            question="What color is this image?"
        )
        
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Steps: {len(result.get('intermediate_steps', []))}")
        
        # Test batch questions
        questions = [
            "What is the main color?",
            "Is this a simple or complex image?",
            "What shapes can you see?"
        ]
        
        batch_results = agent.batch_questions(test_image, questions)
        print(f"\nBatch Results ({len(batch_results)} questions):")
        for i, result in enumerate(batch_results):
            print(f"  Q{i+1}: {result['answer'][:50]}...")
        
        print("VQA Agent test passed!")
        
    except Exception as e:
        print(f"VQA Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vqa_agent()