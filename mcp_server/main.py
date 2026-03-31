from typing import Any, Dict, List

import asyncio
import json
import os
import warnings

# Suppress requests/urllib3/chardet version mismatch warning (cosmetic; from deps)
warnings.filterwarnings(
    "ignore",
    message=".*doesn't match a supported version.*",
)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from neo4j import GraphDatabase
import httpx


class MCPError(Exception):
    pass


def _crawl_content_strings(result) -> tuple:
    """Extract JSON-serializable content from CrawlResult or CrawlResultContainer."""
    # Unwrap CrawlResultContainer to first CrawlResult
    if hasattr(result, "_results") and result._results:
        result = result._results[0]
    md = getattr(result, "markdown", None)
    cleaned = getattr(result, "cleaned_html", None) or ""
    html = getattr(result, "html", None) or ""
    # Crawl4AI can return markdown as MarkdownGenerationResult or StringCompatibleMarkdown
    if md is not None and not isinstance(md, str):
        md_str = getattr(md, "raw_markdown", None) or getattr(md, "fit_markdown", None)
        md = md_str if md_str else (str(md) if md else None)
    return (md or None, cleaned or None, html or None)


async def tool_web_crawl(args: Dict[str, Any]) -> Dict[str, Any]:
    url = args.get("url")
    if not url:
        raise MCPError("url is required")

    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    markdown, cleaned_html, html = _crawl_content_strings(result)
    # Prefer content we can use; fall back to raw html so downstream can strip to text
    success = bool(getattr(result, "success", False))
    status_code = getattr(result, "status_code", None)
    return {
        "success": success,
        "status_code": status_code,
        "markdown": markdown,
        "cleaned_html": cleaned_html,
        "html": html,
    }


async def tool_web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    num_results = int(args.get("num_results", 5))
    if not query:
        raise MCPError("query is required")

    # Placeholder: implement actual search API here.
    # For now, just echo back the query in a structured way.
    return {
        "query": query,
        "results": [],
        "num_results": num_results,
    }


def _get_neo4j_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def _run_neo4j_write(query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    driver = _get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, **parameters)
        return [record.data() for record in result]


def _run_neo4j_read(query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    driver = _get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, **parameters)
        return [record.data() for record in result]


def _ensure_chunk_vector_index(dimensions: int, index_name: str = "chunk_embedding") -> None:
    cypher = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: $dims,
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    _run_neo4j_write(cypher, {"dims": int(dimensions)})


async def tool_graph_upsert_document_chunks(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upsert a Document and its Chunk nodes, store embeddings, and link chunks to entities by name.
    Expected args:
      - doc: {doc_id, company_name, domain, url, ingested_at}
      - chunks: [{chunk_id, text, start, end, embedding, source_url}]
      - chunk_links: [{chunk_id, mention_names: [str]}] optional; per-chunk MENTIONS to (:Entity {name})
      - entity_names: [str] deprecated; if chunk_links omitted, links all chunks to all names (avoid)
      - embedding_dimensions: int required for vector index creation
    """
    doc = args.get("doc") or {}
    chunks = args.get("chunks") or []
    chunk_links = args.get("chunk_links") or []
    entity_names = args.get("entity_names") or []
    dims = args.get("embedding_dimensions")
    if not dims:
        raise MCPError("embedding_dimensions is required")
    if not isinstance(chunks, list):
        raise MCPError("chunks must be a list")

    _ensure_chunk_vector_index(int(dims))

    cypher = """
    MERGE (d:Document {doc_id: $doc_id})
    SET d.company_name = $company_name,
        d.domain = $domain,
        d.url = $url,
        d.ingested_at = $ingested_at
    WITH d
    UNWIND $chunks AS ch
      MERGE (c:Chunk {chunk_id: ch.chunk_id})
      SET c.doc_id = $doc_id,
          c.text = ch.text,
          c.start = ch.start,
          c.end = ch.end,
          c.source_url = ch.source_url,
          c.embedding = ch.embedding
      MERGE (d)-[:HAS_CHUNK]->(c)
    WITH d, collect(c) AS cs
    RETURN d.doc_id AS doc_id, size(cs) AS chunks_upserted
    """

    rows = _run_neo4j_write(
        cypher,
        {
            "doc_id": doc.get("doc_id"),
            "company_name": doc.get("company_name"),
            "domain": doc.get("domain"),
            "url": doc.get("url"),
            "ingested_at": doc.get("ingested_at"),
            "chunks": chunks,
        },
    )

    # Link chunks to entities only when that chunk lists the entity (caller verifies text contains name)
    if chunk_links:
        link_cypher = """
        UNWIND $chunk_links AS row
          MATCH (c:Chunk {chunk_id: row.chunk_id})
          UNWIND row.mention_names AS nm
            WITH c, nm
            WHERE nm IS NOT NULL AND trim(toString(nm)) <> ''
            MATCH (e {name: nm})
            MERGE (c)-[:MENTIONS]->(e)
        RETURN count(*) AS links_attempted
        """
        _run_neo4j_write(link_cypher, {"chunk_links": chunk_links})
    elif entity_names:
        # Legacy: document-wide linking (not recommended)
        link_cypher = """
        UNWIND $chunk_ids AS cid
          MATCH (c:Chunk {chunk_id: cid})
          UNWIND $names AS nm
            MATCH (e {name: nm})
            MERGE (c)-[:MENTIONS]->(e)
        RETURN count(*) AS links_attempted
        """
        _run_neo4j_write(
            link_cypher,
            {
                "chunk_ids": [c.get("chunk_id") for c in chunks if c.get("chunk_id")],
                "names": [n for n in entity_names if n],
            },
        )

    return {"ok": True, "doc_id": (rows[0]["doc_id"] if rows else doc.get("doc_id")), "chunks_upserted": (rows[0]["chunks_upserted"] if rows else 0)}


async def tool_graph_vector_search_chunks(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vector search over Chunk embeddings.
    Args: {embedding: [float], top_k: int}
    """
    embedding = args.get("embedding")
    top_k = int(args.get("top_k", 8))
    if not isinstance(embedding, list) or not embedding:
        raise MCPError("embedding must be a non-empty list")

    cypher = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id,
           node.doc_id AS doc_id,
           node.text AS text,
           node.start AS start,
           node.end AS end,
           score AS score
    """
    rows = _run_neo4j_read(cypher, {"top_k": top_k, "embedding": embedding})
    return {"top_k": top_k, "results": rows}


async def tool_graph_expand_from_chunks(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand from initial chunks to mentioned entities, then traverse entity graph up to `hops`,
    then pull additional chunks mentioning those entities.
    Args: {chunk_ids: [str], hops: int, per_entity_chunks: int}
    """
    chunk_ids = args.get("chunk_ids") or []
    hops = int(args.get("hops", 1))
    per_entity_chunks = int(args.get("per_entity_chunks", 3))
    if not chunk_ids:
        return {"entities": [], "chunks": []}

    entities_cypher = """
    MATCH (c:Chunk)-[:MENTIONS]->(e)
    WHERE c.chunk_id IN $chunk_ids
    RETURN DISTINCT e.name AS name, labels(e) AS labels
    LIMIT 200
    """
    entities = _run_neo4j_read(entities_cypher, {"chunk_ids": chunk_ids})

    # Expand entities through domain graph
    expand_cypher = """
    MATCH (e {name: $name})
    MATCH (e)-[*0..$hops]-(n)
    RETURN DISTINCT n.name AS name, labels(n) AS labels
    LIMIT 400
    """
    expanded = []
    for ent in entities:
        nm = ent.get("name")
        if not nm:
            continue
        expanded.extend(_run_neo4j_read(expand_cypher, {"name": nm, "hops": hops}))

    expanded_names = list({x.get("name") for x in expanded if x.get("name")})

    chunks_cypher = """
    UNWIND $names AS nm
      MATCH (c:Chunk)-[:MENTIONS]->(e {name: nm})
      RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.text AS text, c.start AS start, c.end AS end, nm AS entity
      LIMIT $limit_total
    """
    limit_total = max(10, len(expanded_names) * per_entity_chunks)
    related_chunks = _run_neo4j_read(chunks_cypher, {"names": expanded_names, "limit_total": limit_total})
    return {"entities": expanded, "chunks": related_chunks}

async def tool_graph_upsert_triples(args: Dict[str, Any]) -> Dict[str, Any]:
    triples = args.get("triples") or []
    if not isinstance(triples, list):
        raise MCPError("triples must be a list")

    created = 0
    updated = 0

    for triple in triples:
        subj = triple.get("subject")
        pred = triple.get("predicate")
        obj = triple.get("object")
        rel_type = triple.get("type", "RELATED_TO")
        subj_label = triple.get("subject_label") or "Entity"
        obj_label = triple.get("object_label") or "Entity"

        if not (subj and pred and obj):
            continue

        cypher = f"""
        MERGE (s:{subj_label} {{name: $subject}})
        MERGE (o:{obj_label} {{name: $object}})
        MERGE (s)-[r:{rel_type} {{predicate: $predicate}}]->(o)
        RETURN id(s) AS s_id, id(o) AS o_id, id(r) AS r_id
        """

        _run_neo4j_write(
            cypher,
            {"subject": subj, "predicate": pred, "object": obj},
        )
        created += 1

    return {"created_or_merged_triples": created, "updated_triples": updated}


async def tool_graph_search_paths(args: Dict[str, Any]) -> Dict[str, Any]:
    source_company = args.get("source_company")
    target_company = args.get("target_company")
    strategy = args.get("strategy", "shortest_path")

    if not (source_company and target_company):
        raise MCPError("source_company and target_company are required")

    if strategy == "shortest_path":
        cypher = """
        MATCH (a:Company {name: $source}), (b:Company {name: $target}),
        p = shortestPath((a)-[*..5]-(b))
        RETURN p
        """
    elif strategy == "shared_board_members":
        cypher = """
        MATCH (a:Company {name: $source})<-[:BOARD_MEMBER_OF]-(p:Person)-[:BOARD_MEMBER_OF]->(b:Company {name: $target})
        RETURN a, p, b
        """
    else:
        raise MCPError(f"Unsupported strategy: {strategy}")

    driver = _get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            cypher, source=source_company, target=target_company
        )
        records = [record.data() for record in result]

    return {"strategy": strategy, "results": records}


async def tool_graph_apply_wikidata_linking(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store Wikidata entity linking on the anchor company and merge related companies
    from curated Wikidata claims (parent / subsidiary only).

    Args:
      anchor_name: Company.name in the graph to attach wikidata props to
      wikidata_id: e.g. "Q312"
      wikidata_label: preferred English label
      wikidata_uri: optional https://www.wikidata.org/wiki/Q...
      related: list of { rel_type, target_id, target_label }
        rel_type in: HAS_PARENT_ORGANIZATION | HAS_SUBSIDIARY
    """
    anchor_name = args.get("anchor_name")
    wikidata_id = args.get("wikidata_id")
    wikidata_label = args.get("wikidata_label")
    wikidata_uri = args.get("wikidata_uri")
    related = args.get("related") or []

    if not (anchor_name and wikidata_id):
        raise MCPError("anchor_name and wikidata_id are required")

    _run_neo4j_write(
        """
        MERGE (c:Company {name: $anchor_name})
        SET c.wikidata_id = $wikidata_id,
            c.wikidata_label = $wikidata_label,
            c.wikidata_uri = $wikidata_uri
        """,
        {
            "anchor_name": anchor_name,
            "wikidata_id": wikidata_id,
            "wikidata_label": wikidata_label,
            "wikidata_uri": wikidata_uri or f"https://www.wikidata.org/wiki/{wikidata_id}",
        },
    )

    allowed = {"HAS_PARENT_ORGANIZATION", "HAS_SUBSIDIARY"}
    applied = 0
    for rel in related:
        rt = rel.get("rel_type")
        tid = rel.get("target_id")
        tlabel = rel.get("target_label") or tid
        if rt not in allowed or not tid:
            continue
        if rt == "HAS_PARENT_ORGANIZATION":
            cypher = """
            MATCH (child:Company {name: $anchor_name})
            MERGE (parent:Company {wikidata_id: $tid})
            ON CREATE SET parent.name = $tlabel
            MERGE (child)-[:HAS_PARENT_ORGANIZATION {source: 'wikidata'}]->(parent)
            """
        else:
            cypher = """
            MATCH (parent:Company {name: $anchor_name})
            MERGE (sub:Company {wikidata_id: $tid})
            ON CREATE SET sub.name = $tlabel
            MERGE (parent)-[:HAS_SUBSIDIARY {source: 'wikidata'}]->(sub)
            """
        _run_neo4j_write(
            cypher,
            {"anchor_name": anchor_name, "tid": tid, "tlabel": tlabel},
        )
        applied += 1

    return {"ok": True, "wikidata_id": wikidata_id, "related_edges_applied": applied}


TOOL_REGISTRY = {
    "web_crawl": tool_web_crawl,
    "web_search": tool_web_search,
    "graph_upsert_triples": tool_graph_upsert_triples,
    "graph_search_paths": tool_graph_search_paths,
    "graph_upsert_document_chunks": tool_graph_upsert_document_chunks,
    "graph_vector_search_chunks": tool_graph_vector_search_chunks,
    "graph_expand_from_chunks": tool_graph_expand_from_chunks,
    "graph_apply_wikidata_linking": tool_graph_apply_wikidata_linking,
}


async def handle_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise MCPError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]
    return await tool(args)


async def main():
    """
    Minimal CLI entrypoint for local testing.
    In a real MCP server, this would be wired into the MCP transport.
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: main.py TOOL_NAME JSON_ARGS | main.py TOOL_NAME -  # JSON on stdin")
        raise SystemExit(1)

    tool_name = sys.argv[1]
    payload = sys.argv[2]
    if payload == "-":
        args = json.load(sys.stdin)
    else:
        args = json.loads(payload)

    # Crawl4AI logs to stdout; redirect so only our JSON goes to stdout for the client.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = await handle_tool_call(tool_name, args)
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())

