from typing import TypedDict, List, Optional, Annotated, Literal, Any

from langgraph.graph.message import add_messages


GoalType = Literal["ingestion", "query"]


class LeadState(TypedDict, total=False):
    # Conversation and core identifiers
    messages: Annotated[list, add_messages]
    company_name: str
    domain: str
    question: Optional[str]

    # High-level mode and planning
    goal_type: GoalType  # e.g. "ingestion" or "query"
    plan_steps: List[dict]  # structured steps from Planner
    current_step: Optional[int]

    # Research / web content
    raw_web_content: Optional[str]
    plain_text_for_extraction: Optional[str]  # cleaned text actually sent to the LLM (ingestion)

    # Lexical layer (chunks/embeddings)
    doc_id: Optional[str]
    chunks_created: Optional[int]
    chunks: Optional[List[dict]]  # debug: chunk_id/text/start/end
    retrieved_chunks: Optional[List[dict]]  # debug: from vector search + expansion
    retrieved_entities: Optional[List[dict]]  # debug: entities from expansion

    # Graph ingestion + retrieval
    extracted_triples: List[dict]
    graph_results: Optional[Any]
    graph_report: Optional[str]
    wikidata_enrichment: Optional[dict]  # resolved Q-id + edges from KB (ingestion)

    # Control and status
    error_status: Optional[int]  # e.g. HTTP 403 from web crawl
    crawl_error: Optional[str]  # MCP/subprocess error when crawl returns no content
    final_report: Optional[str]

