"""
LLM generation with fallback chains.

Handles calling LLMs with cost tracking and error handling.
"""

from typing import List, Tuple, Optional
import time
import structlog
from langchain.chat_models import ChatGroq
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage, SystemMessage

logger = structlog.get_logger()

class LLMManager:
    """
    Manages LLM calls with fallback and cost tracking.
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize LLM manager.
        
        Args:
            groq_api_key: Groq API key
        """
        self.groq_api_key = groq_api_key
        self.primary_llm = self._init_groq()
        self.request_count = 0
        self.total_cost = 0.0
    
    def _init_groq(self):
        """Initialize Groq LLM."""
        return ChatGroq(
            temperature=0.3,
            groq_api_key=self.groq_api_key,
            model_name="mixtral-8x7b-32768",
        )
    
    def generate(
        self,
        query: str,
        context_docs: List[str],
    ) -> Tuple[str, float, int, int]:
        """
        Generate answer from query and context.
        
        Args:
            query: User query
            context_docs: List of relevant document texts
            
        Returns:
            (answer, cost_usd, prompt_tokens, completion_tokens)
        """
        start_time = time.time()
        
        # Format context
        context_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc[:500]}"  # First 500 chars per doc
            for i, doc in enumerate(context_docs)
        ])
        
        # Create prompt
        system_msg = SystemMessage(content="""
You are a helpful assistant. Answer the user's question based ONLY on the provided context.

If the answer is not in the context, say "I cannot find this information in the provided documents."

Always cite which document(s) you're using for your answer using [Document X] format.
""")
        
        user_msg = HumanMessage(content=f"""
Context:
{context_text}

Question: {query}

Answer:
""")
        
        try:
            # Call LLM
            response = self.primary_llm.invoke([system_msg, user_msg])
            answer = response.content
            
            # Estimate tokens (roughly: 1 word ≈ 1.3 tokens)
            prompt_tokens = int(len((system_msg.content + user_msg.content).split()) * 1.3)
            completion_tokens = int(len(answer.split()) * 1.3)
            
            # Groq pricing: $0.27 per 1M input tokens, $0.27 per 1M output tokens
            cost = (prompt_tokens * 0.27 + completion_tokens * 0.27) / 1_000_000
            
            self.request_count += 1
            self.total_cost += cost
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "LLM generation complete",
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )
            
            return answer, cost, prompt_tokens, completion_tokens
        
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            
            # Fallback: return simple concatenation
            answer = f"Unable to generate answer. Error: {str(e)}"
            return answer, 0.0, 0, 0

def get_llm_manager(groq_api_key: str) -> LLMManager:
    """Create LLM manager."""
    return LLMManager(groq_api_key)
