"""
LLM generation.

Adds:
- Per-intent system instruction override (from router)
- Embedding cache integration
- Slightly cleaner cost tracking
"""

from typing import List, Tuple, Optional
import time
import structlog
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings

logger = structlog.get_logger()

_BASE_SYSTEM = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.

If the answer is not in the context, say "I cannot find this information in the provided documents."

Always cite which document(s) you're using for your answer using [Document X] format."""


class LLMManager:
    """Manages LLM calls with cost tracking."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=groq_api_key,
            model_name=settings.GENERATION_MODEL,
        )
        self.request_count = 0
        self.total_cost = 0.0

    def generate(
        self,
        query: str,
        context_docs: List[str],
        system_instruction: str = "",
    ) -> Tuple[str, float, int, int]:
        """
        Generate answer from query and context.

        Args:
            query: User query
            context_docs: List of relevant document texts
            system_instruction: Additional instruction from router (intent-specific)

        Returns:
            (answer, cost_usd, prompt_tokens, completion_tokens)
        """
        start = time.time()

        context_text = "\n\n".join(
            f"[Document {i+1}]:\n{doc[:600]}"
            for i, doc in enumerate(context_docs)
        )

        system_content = _BASE_SYSTEM
        if system_instruction:
            system_content += f"\n\nAdditional guidance: {system_instruction}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content

            prompt_tokens = int(len((system_content + context_text + query).split()) * 1.3)
            completion_tokens = int(len(answer.split()) * 1.3)

            # Groq mixtral pricing (approximate)
            cost = (prompt_tokens * 0.27 + completion_tokens * 0.27) / 1_000_000

            self.request_count += 1
            self.total_cost += cost

            logger.info(
                "LLM generation complete",
                latency_ms=int((time.time() - start) * 1000),
                cost_usd=round(cost, 6),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            return answer, cost, prompt_tokens, completion_tokens

        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return f"Generation error: {str(e)}", 0.0, 0, 0


_llm_manager: Optional[LLMManager] = None


def get_llm_manager(groq_api_key: Optional[str] = None) -> LLMManager:
    """Get or create LLM manager (singleton)."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager(groq_api_key or settings.GROQ_API_KEY)
    return _llm_manager