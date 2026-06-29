"""
Multi-agent router for query classification and routing.

Classifies incoming queries and routes them to the appropriate
retrieval strategy or agent.
"""

from enum import Enum
from typing import TypedDict, Annotated, Sequence, Optional
import operator
import structlog

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    FACTUAL = "factual"          # "What is the termination clause?"
    COMPARATIVE = "comparative"  # "Compare policy A vs policy B"
    SUMMARIZE = "summarize"      # "Summarize this contract"
    ANALYTICAL = "analytical"    # "What are the risks in this agreement?"
    UNKNOWN = "unknown"


class AgentState(TypedDict):
    """State passed between agents in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    intent: Optional[QueryIntent]
    retrieved_chunks: Optional[list]
    reranked_chunks: Optional[list]
    answer: Optional[str]
    cost_usd: float
    latency_ms: int
    error: Optional[str]


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies based on intent.

    Uses an LLM to classify intent, then routes to:
    - Factual: standard hybrid retrieval
    - Comparative: multi-doc retrieval
    - Summarize: full-doc retrieval
    - Analytical: chain-of-thought + retrieval
    """

    INTENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier. Classify the user's query into exactly one of these intents:

- factual: asking for a specific fact or detail
- comparative: comparing two or more things
- summarize: asking for a summary or overview
- analytical: asking for analysis, risks, or implications
- unknown: cannot determine intent

Respond with ONLY the intent word, nothing else."""),
        ("human", "{query}"),
    ])

    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
            model_name="mixtral-8x7b-32768",
        )
        self.classifier = self.INTENT_PROMPT | self.llm
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("route", self._route_node)
        graph.add_node("factual_retrieve", self._factual_retrieve_node)
        graph.add_node("comparative_retrieve", self._comparative_retrieve_node)
        graph.add_node("summarize_retrieve", self._summarize_retrieve_node)
        graph.add_node("analytical_retrieve", self._analytical_retrieve_node)
        graph.add_node("generate", self._generate_node)

        # Edges
        graph.set_entry_point("classify")
        graph.add_edge("classify", "route")

        graph.add_conditional_edges(
            "route",
            self._dispatch,
            {
                QueryIntent.FACTUAL: "factual_retrieve",
                QueryIntent.COMPARATIVE: "comparative_retrieve",
                QueryIntent.SUMMARIZE: "summarize_retrieve",
                QueryIntent.ANALYTICAL: "analytical_retrieve",
                QueryIntent.UNKNOWN: "factual_retrieve",  # default
            },
        )

        for node in ["factual_retrieve", "comparative_retrieve",
                     "summarize_retrieve", "analytical_retrieve"]:
            graph.add_edge(node, "generate")

        graph.add_edge("generate", END)

        return graph.compile()

    async def run(self, query: str, db) -> AgentState:
        """
        Run the agent graph for a query.

        Args:
            query: User query
            db: Database session

        Returns:
            Final agent state with answer
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "answer": None,
            "cost_usd": 0.0,
            "latency_ms": 0,
            "error": None,
        }

        # Store db on instance for nodes to use
        self._db = db

        result = await self._graph.ainvoke(initial_state)
        return result

    # --- Node implementations ---

    async def _classify_node(self, state: AgentState) -> AgentState:
        """Classify query intent."""
        try:
            response = await self.classifier.ainvoke({"query": state["query"]})
            intent_str = response.content.strip().lower()
            intent = QueryIntent(intent_str) if intent_str in QueryIntent._value2member_map_ else QueryIntent.UNKNOWN

            logger.info("Query classified", query=state["query"][:50], intent=intent)
            return {**state, "intent": intent}

        except Exception as e:
            logger.error("Classification failed", error=str(e))
            return {**state, "intent": QueryIntent.UNKNOWN, "error": str(e)}

    async def _route_node(self, state: AgentState) -> AgentState:
        """Routing node (pass-through; routing done via conditional edges)."""
        return state

    def _dispatch(self, state: AgentState) -> str:
        """Return the intent value for conditional edge dispatch."""
        return state.get("intent", QueryIntent.UNKNOWN)

    async def _factual_retrieve_node(self, state: AgentState) -> AgentState:
        """Standard hybrid retrieval for factual queries."""
        from app.retriever import get_retriever
        from app.reranker import get_reranker

        retriever = get_retriever(self._db)
        chunks = await retriever.retrieve(state["query"], top_k=settings.DENSE_TOP_K)

        reranker = get_reranker()
        reranked = reranker.rerank(state["query"], chunks, top_k=settings.FINAL_TOP_K)

        return {**state, "retrieved_chunks": chunks, "reranked_chunks": reranked}

    async def _comparative_retrieve_node(self, state: AgentState) -> AgentState:
        """
        Multi-doc retrieval for comparative queries.
        Retrieves more documents to capture both sides.
        """
        from app.retriever import get_retriever
        from app.reranker import get_reranker

        retriever = get_retriever(self._db)
        # Over-retrieve to get docs from multiple sources
        chunks = await retriever.retrieve(state["query"], top_k=settings.DENSE_TOP_K * 2)

        reranker = get_reranker()
        # Keep more context for comparison
        reranked = reranker.rerank(state["query"], chunks, top_k=settings.FINAL_TOP_K * 2)

        return {**state, "retrieved_chunks": chunks, "reranked_chunks": reranked}

    async def _summarize_retrieve_node(self, state: AgentState) -> AgentState:
        """
        Full-document retrieval for summarization queries.
        Retrieves chunks ordered by document and position.
        """
        from app.retriever import get_retriever

        retriever = get_retriever(self._db)
        # Dense-only retrieval, ordered by doc position (not relevance)
        chunks = await retriever.retrieve(
            state["query"],
            top_k=settings.DENSE_TOP_K * 2,
            use_sparse=False,  # Don't use BM25 for summarization
        )

        # Sort by document_id and chunk_index for coherent summaries
        chunks.sort(key=lambda x: (x.get("doc_id", ""), x.get("chunk_index", 0)))

        return {**state, "retrieved_chunks": chunks, "reranked_chunks": chunks}

    