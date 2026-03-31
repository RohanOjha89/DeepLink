import re
import uuid
from html import unescape
from typing import Dict, Any, List, Optional

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from state import LeadState
from mcp_client import call_mcp_tool
from graph_models import OrganizationGraph, Location, Person, Product, Subsidiary, Industry


client = AsyncOpenAI()

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"


async def _wikidata_get(params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(
        timeout=45.0,
        headers={"User-Agent": "LeadGen/1.0 (https://example.local; research bot)"},
    ) as client_http:
        r = await client_http.get(_WIKIDATA_API, params=params)
        r.raise_for_status()
        return r.json()


def _wikidata_claim_qids(entity: Dict[str, Any], pid: str) -> List[str]:
    out: List[str] = []
    for stmt in entity.get("claims", {}).get(pid, []):
        dv = stmt.get("mainsnak", {}).get("datavalue")
        if not dv or dv.get("type") != "wikibase-entityid":
            continue
        q = dv.get("value", {}).get("id")
        if q:
            out.append(q)
    return out


async def _wikidata_enrich_company(anchor_name: str) -> Optional[Dict[str, Any]]:
    """
    Entity linking: resolve company name to a Wikidata item and pull a few
    structural company-company relations (parent / subsidiary) from statements.
    Does NOT infer supply-chain (e.g. TSMC supplier of Apple) unless Wikidata encodes it in
    a property we import (we only import P749, P355 here).
    """
    if not anchor_name or anchor_name.strip() in ("", "Unknown"):
        return None

    search = await _wikidata_get(
        {
            "action": "wbsearchentities",
            "search": anchor_name.strip(),
            "language": "en",
            "format": "json",
            "limit": "8",
        }
    )
    hits = search.get("search") or []
    if not hits:
        return None

    an = anchor_name.strip().lower()
    best = None
    for h in hits:
        if (h.get("label") or "").strip().lower() == an:
            best = h
            break
    if best is None:
        best = hits[0]

    qid = best.get("id")
    if not qid:
        return None

    blob = await _wikidata_get(
        {
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels|claims",
            "languages": "en",
            "format": "json",
        }
    )
    entity = (blob.get("entities") or {}).get(qid)
    if not entity:
        return None

    label_en = (entity.get("labels") or {}).get("en", {}).get("value")
    wikidata_label = label_en or best.get("label") or anchor_name

    p749 = _wikidata_claim_qids(entity, "P749")
    p355 = _wikidata_claim_qids(entity, "P355")
    all_related = list({*p749, *p355})

    labels_map: Dict[str, str] = {qid: wikidata_label}
    if all_related:
        labels_blob = await _wikidata_get(
            {
                "action": "wbgetentities",
                "ids": "|".join(all_related),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
        )
        for tid, tobj in (labels_blob.get("entities") or {}).items():
            lbl = (tobj.get("labels") or {}).get("en", {}).get("value")
            if lbl:
                labels_map[tid] = lbl

    related: List[Dict[str, str]] = []
    for tid in p749:
        related.append(
            {
                "rel_type": "HAS_PARENT_ORGANIZATION",
                "target_id": tid,
                "target_label": labels_map.get(tid, tid),
            }
        )
    for tid in p355:
        related.append(
            {
                "rel_type": "HAS_SUBSIDIARY",
                "target_id": tid,
                "target_label": labels_map.get(tid, tid),
            }
        )

    return {
        "anchor_name": anchor_name.strip(),
        "wikidata_id": qid,
        "wikidata_label": wikidata_label,
        "wikidata_uri": f"https://www.wikidata.org/wiki/{qid}",
        "related": related,
    }


async def _apply_wikidata_linking(anchor_name: str) -> Dict[str, Any]:
    """Resolve anchor to Wikidata and persist linking + structural edges via MCP."""
    try:
        wk = await _wikidata_enrich_company(anchor_name or "")
    except Exception as exc:
        return {"resolved": None, "mcp": None, "error": repr(exc)}
    if not wk:
        return {"resolved": None, "mcp": None}
    try:
        mcp = await call_mcp_tool("graph_apply_wikidata_linking", wk)
    except Exception as exc:
        return {"resolved": wk, "mcp": None, "error": repr(exc)}
    return {"resolved": wk, "mcp": mcp}


def _openai_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize schema for OpenAI: additionalProperties false, required array. No extra keywords alongside $ref."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if "$ref" in out:
        return out
    if "description" not in out:
        out["description"] = ""
    if out.get("type") == "object":
        out["additionalProperties"] = False
        if "properties" in out and isinstance(out["properties"], dict):
            props = {}
            for k, v in out["properties"].items():
                vnorm = _openai_schema(v)
                if isinstance(vnorm, dict) and "$ref" not in vnorm and vnorm.get("description") == "":
                    vnorm["description"] = k.replace("_", " ")
                props[k] = vnorm
            out["properties"] = props
            required = [k for k in out["properties"] if not (isinstance(out["properties"][k], dict) and out["properties"][k].get("$ref"))]
            out["required"] = required if required else list(out["properties"].keys())
    if "items" in out:
        it = out["items"]
        out["items"] = _openai_schema(it) if isinstance(it, dict) else it
    for key in ("oneOf", "anyOf", "allOf"):
        if key in out and isinstance(out[key], list):
            out[key] = [_openai_schema(x) for x in out[key]]
    if "$defs" in out and isinstance(out["$defs"], dict):
        out["$defs"] = {k: _openai_schema(v) for k, v in out["$defs"].items()}
    return out


def _html_to_plain_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace so the LLM sees readable text."""
    if not html or "<" not in html or ">" not in html:
        return html
    # Remove script/style and their contents
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    # Remove SVG (icons/graphics add no extractable text, just noise)
    text = re.sub(r"<svg[^>]*>[\s\S]*?</svg>", " ", text, flags=re.IGNORECASE)
    # Remove HTML comments (e.g. "Font Awesome fontawesome.com")
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)
    # Strip Wikipedia-style references and citations so we keep article body, not refs
    text = re.sub(r"<ol[^>]*class=\"references\"[^>]*>[\s\S]*?</ol>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<div[^>]*class=\"reflist\"[^>]*>[\s\S]*?</div>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\s+id=\"cite_note-\d+\"[\s\S]*?</li>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Strip any span with citation/reference class (e.g. reference-text, citation web)
    text = re.sub(r"<span[^>]*class=\"[^\"]*reference-text[^\"]*\"[^>]*>[\s\S]*?</span>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<cite[^>]*class=\"[^\"]*citation[^\"]*\"[^>]*>[\s\S]*?</cite>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Preserve img alt text (e.g. "Three people excitedly considering...") before stripping tags
    text = re.sub(
        r"<img[^>]*alt=([\"'])([^\1]*?)\1[^>]*>",
        r" \2 ",
        text,
        flags=re.IGNORECASE,
    )
    # Replace remaining tags with space to avoid joining words
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    # Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _company_core_for_match(name: str) -> str:
    """Lowercase name with trailing legal suffixes removed (e.g. 'Apple Inc.' -> 'apple')."""
    s = name.strip().lower()
    while s:
        prev = s
        for suf in (
            ", inc.",
            " inc.",
            ", inc",
            " inc",
            ", ltd.",
            " ltd.",
            ", ltd",
            " ltd",
            ", corp.",
            " corp.",
            ", corp",
            " corp",
            " plc",
            " llc",
            ", llc",
            " limited",
            " company",
            ", company",
        ):
            if s.endswith(suf):
                s = s[: -len(suf)].strip()
                break
        if s == prev:
            break
    return s.strip()


def _entity_mentioned_in_chunk(chunk_lower: str, entity_name: str) -> bool:
    """
    Decide if a chunk text refers to an extracted entity name, without requiring
    exact string match (e.g. page says 'Apple' but the graph has 'Apple Inc.').
    This is lightweight lexical matching—not a semantic/embedding resolver.
    """
    tl = chunk_lower
    nm = (entity_name or "").strip()
    if not nm:
        return False
    el = nm.lower()
    if el in tl:
        return True
    core = _company_core_for_match(nm)
    if core and len(core) >= 2 and core in tl:
        return True
    # Significant tokens (avoid tiny words like 'of', 'the')
    for part in re.split(r"[\s,.]+", el):
        if len(part) >= 4 and part in tl:
            return True
    # Short uppercase-style tickers / brands (TSMC, AMD, …)
    raw = nm.strip()
    if 2 <= len(raw) <= 6 and raw.isalpha() and raw.upper() == raw:
        return raw.lower() in tl
    return False


def _chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[dict]:
    """
    Simple chunker (char-based) to avoid extra deps.
    Returns [{chunk_id, text, start, end}].
    """
    if not text:
        return []
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    chunks: List[dict] = []
    i = 0
    n = len(t)
    max_chars = max(200, int(max_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chars - 50))
    while i < n:
        j = min(n, i + max_chars)
        chunk = t[i:j].strip()
        if chunk:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk,
                    "start": i,
                    "end": j,
                }
            )
        if j >= n:
            break
        i = max(0, j - overlap_chars)
    return chunks


async def _embed_texts_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    resp = await client.embeddings.create(model=model, input=texts)
    # resp.data is in the same order as input
    return [d.embedding for d in resp.data]


class PlanStep(BaseModel):
    name: str
    description: str | None = None


class PlanOutput(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    # semantic relation label used as edge type in Neo4j MCP tool
    type: str = "RELATED_TO"
    subject_label: str | None = Field(
        default=None, description="Optional Neo4j label for subject node"
    )
    object_label: str | None = Field(
        default=None, description="Optional Neo4j label for object node"
    )


class TripleExtraction(BaseModel):
    triples: List[Triple] = Field(default_factory=list)


def _location_to_string(loc: Location) -> str:
    parts = [p for p in [loc.city, loc.state, loc.country] if p]
    return ", ".join(parts) if parts else ""


def organization_to_triples(org: OrganizationGraph) -> List[Triple]:
    """
    Convert a rich OrganizationGraph into a flat list of typed triples
    suitable for Neo4j ingestion via the MCP tool.
    """
    triples: List[Triple] = []
    org_name = org.organization_name

    # Core attributes as simple triples
    if org.description:
        triples.append(
            Triple(
                subject=org_name,
                predicate="hasDescription",
                object=org.description,
                type="HAS_DESCRIPTION",
                subject_label="Company",
                object_label="Literal",
            )
        )
    if org.founded_year:
        triples.append(
            Triple(
                subject=org_name,
                predicate="foundedIn",
                object=org.founded_year,
                type="FOUNDED_IN",
                subject_label="Company",
                object_label="Literal",
            )
        )
    if org.employee_count:
        triples.append(
            Triple(
                subject=org_name,
                predicate="hasEmployeeCount",
                object=org.employee_count,
                type="HAS_EMPLOYEE_COUNT",
                subject_label="Company",
                object_label="Literal",
            )
        )
    if org.stock_ticker:
        triples.append(
            Triple(
                subject=org_name,
                predicate="hasStockTicker",
                object=org.stock_ticker,
                type="HAS_STOCK_TICKER",
                subject_label="Company",
                object_label="Literal",
            )
        )

    # Headquarters
    if org.headquarters:
        loc_str = _location_to_string(org.headquarters)
        if loc_str:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="hasHeadquarters",
                    object=loc_str,
                    type="HAS_HEADQUARTERS",
                    subject_label="Company",
                    object_label="Location",
                )
            )

    # Operating locations
    if org.operating_locations:
        for loc in org.operating_locations:
            loc_str = _location_to_string(loc)
            if loc_str:
                triples.append(
                    Triple(
                        subject=org_name,
                        predicate="operatesIn",
                        object=loc_str,
                        type="OPERATES_IN",
                        subject_label="Company",
                        object_label="Location",
                    )
                )

    # Key people
    if org.key_people:
        for person in org.key_people:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="hasPerson",
                    object=person.name,
                    type="HAS_PERSON",
                    subject_label="Company",
                    object_label="Person",
                )
            )

    # Products / services
    if org.products_services:
        for product in org.products_services:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="offers",
                    object=product.name,
                    type="OFFERS",
                    subject_label="Company",
                    object_label="Product",
                )
            )

    # Industries
    if org.industries:
        for industry in org.industries:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="belongsTo",
                    object=industry.name,
                    type="BELONGS_TO",
                    subject_label="Company",
                    object_label="Industry",
                )
            )

    # Subsidiaries
    if org.subsidiaries:
        for sub in org.subsidiaries:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="hasSubsidiary",
                    object=sub.name,
                    type="HAS_SUBSIDIARY",
                    subject_label="Company",
                    object_label="Company",
                )
            )

    # Parent company
    if org.parent_company:
        triples.append(
            Triple(
                subject=org_name,
                predicate="isSubsidiaryOf",
                object=org.parent_company,
                type="IS_SUBSIDIARY_OF",
                subject_label="Company",
                object_label="Company",
            )
        )

    # Partners
    if org.partners:
        for partner in org.partners:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="partnersWith",
                    object=partner,
                    type="PARTNERS_WITH",
                    subject_label="Company",
                    object_label="Company",
                )
            )

    # Competitors
    if org.competitors:
        for competitor in org.competitors:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="competesAgainst",
                    object=competitor,
                    type="COMPETES_AGAINST",
                    subject_label="Company",
                    object_label="Company",
                )
            )

    # Customer / supplier (inter-company edges from page text)
    if org.major_customers:
        for cust in org.major_customers:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="hasMajorCustomer",
                    object=cust,
                    type="HAS_MAJOR_CUSTOMER",
                    subject_label="Company",
                    object_label="Company",
                )
            )
    if org.key_suppliers:
        for sup in org.key_suppliers:
            triples.append(
                Triple(
                    subject=org_name,
                    predicate="hasSupplier",
                    object=sup,
                    type="HAS_SUPPLIER",
                    subject_label="Company",
                    object_label="Company",
                )
            )

    return triples


def _normalize_url(domain: str) -> str:
    if domain.startswith("http://") or domain.startswith("https://"):
        return domain
    return f"https://{domain}"


async def planner_agent(state: LeadState) -> Dict[str, Any]:
    """
    Uses an LLM to create a simple, structured plan for ingestion or query.
    """
    goal_type = state.get("goal_type", "ingestion")
    company_name = state.get("company_name", "")
    domain = state.get("domain", "")
    question = state.get("question", "")

    system_prompt = (
        "You are a planning agent for a lead intelligence system. "
        "Return a JSON list of high-level steps for the given goal."
    )

    user_prompt = f"Goal type: {goal_type}\nCompany: {company_name}\nDomain: {domain}\nQuestion: {question}\n"

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "plan_output",
                "strict": True,
                "schema": _openai_schema(PlanOutput.model_json_schema()),
            },
        },
    )

    content = response.choices[0].message.content or "{}"
    try:
        plan_obj = PlanOutput.model_validate_json(content)
    except ValidationError:
        plan_obj = PlanOutput()

    plan_steps: List[dict] = [step.model_dump() for step in plan_obj.steps]
    return {"plan_steps": plan_steps, "current_step": 0}


async def researcher_agent(state: LeadState) -> Dict[str, Any]:
    """
    Ingestion: always web_crawl the given URL or https://domain so behavior matches
    across companies. Routing to web_search returned JSON blobs that skipped LLM
    extraction. On HTTP 403 only, fall back to web_search.
    """
    domain = state.get("domain") or state.get("company_name")
    if not domain:
        return {"error_status": 400}

    url = _normalize_url(domain)
    tool_name = "web_crawl"
    result = await call_mcp_tool("web_crawl", {"url": url})

    if not result.get("success") and result.get("status_code") == 403:
        search_result = await call_mcp_tool(
            "web_search", {"query": domain, "num_results": 8}
        )
        return {
            "raw_web_content": __import__("json").dumps(search_result),
            "error_status": 403,
            "crawl_error": "web_crawl returned 403; used web_search fallback",
        }

    content = (
        result.get("markdown")
        or result.get("cleaned_html")
        or result.get("html")
    )

    crawl_error = None
    if not content:
        if not result.get("success") and result.get("error"):
            crawl_error = result.get("error", "").strip() or "Crawl subprocess failed"
        elif tool_name == "web_crawl":
            crawl_error = (
                f"Crawl returned no markdown/html (success={result.get('success')}, "
                f"status_code={result.get('status_code')})"
            )

    return {
        "raw_web_content": content,
        "error_status": result.get("status_code"),
        "crawl_error": crawl_error,
    }



async def graph_architect_agent(state: LeadState) -> Dict[str, Any]:
    """
    Ingestion mode: extract triples from raw_web_content and upsert into Neo4j via MCP.
    Query mode: run a graph search and return serialized results.
    """
    goal_type = state.get("goal_type", "ingestion")
    raw_web_content = state.get("raw_web_content") or ""
    question = state.get("question") or ""
    company_name = state.get("company_name") or ""

    if goal_type == "ingestion":
        domain = state.get("domain") or ""
        minimal_org = OrganizationGraph(
            organization_name=company_name or "Unknown",
            domain=domain,
        )
        raw_stripped = (raw_web_content or "").strip()
        is_likely_search_fallback = (
            raw_stripped.startswith("{")
            and ("query" in raw_stripped and "results" in raw_stripped)
        )
        # Convert HTML to plain text so length check and LLM see readable content
        plain_text = (_html_to_plain_text(raw_stripped) or raw_stripped).strip()
        # Keep thresholds modest so Wikipedia / mobile HTML still extracts; search
        # JSON blob is still excluded via is_likely_search_fallback.
        has_enough_content = (
            len(raw_stripped) >= 200
            and len(plain_text) >= 200
            and not is_likely_search_fallback
        )

        if not has_enough_content:
            triples = [t.model_dump() for t in organization_to_triples(minimal_org)]
            upsert_result = await call_mcp_tool(
                "graph_upsert_triples", {"triples": triples}
            )
            wk_bundle = await _apply_wikidata_linking(
                company_name or minimal_org.organization_name or ""
            )
            report = {
                **upsert_result,
                "skipped_extraction": True,
                "reason": "crawl_returned_no_usable_content",
                "wikidata": wk_bundle,
            }
            return {
                "extracted_triples": triples,
                "graph_report": __import__("json").dumps(report),
                "plain_text_for_extraction": plain_text[:8000],
                "wikidata_enrichment": wk_bundle,
            }

        system_prompt = (
            "You are an information extraction agent. "
            "Extract ONLY information that is explicitly stated in the text below. "
            "Do NOT invent, assume, or guess any names, products, people, or connections. "
            "If the text does not mention something, leave that field null/empty. "
            "Use the exact organization name and domain from the page when possible. "
            "When the text clearly names other companies as major customers (buyers of "
            "your output) or as suppliers to the subject company, list them in "
            "major_customers or key_suppliers so company-to-company edges can be built."
        )
        user_prompt = plain_text[:8000]

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "organization_graph",
                    "strict": True,
                    "schema": _openai_schema(OrganizationGraph.model_json_schema()),
                },
            },
        )

        content = response.choices[0].message.content or "{}"
        try:
            org = OrganizationGraph.model_validate_json(content)
        except ValidationError:
            org = minimal_org
        if not org.organization_name or org.organization_name == "Unknown":
            org = OrganizationGraph(
                organization_name=company_name or "Unknown",
                domain=org.domain or domain,
                description=org.description,
                founded_year=org.founded_year,
                employee_count=org.employee_count,
                stock_ticker=org.stock_ticker,
                headquarters=org.headquarters,
                operating_locations=org.operating_locations,
                key_people=org.key_people,
                products_services=org.products_services,
                industries=org.industries,
                subsidiaries=org.subsidiaries,
                parent_company=org.parent_company,
                partners=org.partners,
                competitors=org.competitors,
                major_customers=org.major_customers,
                key_suppliers=org.key_suppliers,
            )

        triples = [t.model_dump() for t in organization_to_triples(org)]
        upsert_result = await call_mcp_tool(
            "graph_upsert_triples", {"triples": triples}
        )
        wk_bundle = await _apply_wikidata_linking(
            org.organization_name or company_name or ""
        )

        # Lexical layer: chunk + embed + upsert chunk nodes and link to extracted entities
        doc_id = str(uuid.uuid4())
        chunks = _chunk_text(plain_text, max_chars=1200, overlap_chars=150)
        embeddings = await _embed_texts_openai([c["text"] for c in chunks])
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb
            c["source_url"] = domain

        entity_names: List[str] = []
        anchor = (company_name or "").strip()
        if anchor:
            entity_names.append(anchor)
        for t in triples:
            s = t.get("subject")
            o = t.get("object")
            if s:
                entity_names.append(s)
            if o:
                entity_names.append(o)
        # de-dupe, preserve order
        seen = set()
        entity_names = [x for x in entity_names if not (x in seen or seen.add(x))]

        # MENTIONS when chunk text matches entity (exact, core name, or token/symbol heuristics).
        chunk_links: List[dict] = []
        for c in chunks:
            tl = (c.get("text") or "").lower()
            mention_names = [
                nm for nm in entity_names if nm and _entity_mentioned_in_chunk(tl, nm)
            ]
            chunk_links.append({"chunk_id": c["chunk_id"], "mention_names": mention_names})

        chunks_result = await call_mcp_tool(
            "graph_upsert_document_chunks",
            {
                "doc": {
                    "doc_id": doc_id,
                    "company_name": company_name,
                    "domain": domain,
                    "url": domain,
                    "ingested_at": __import__("time").time(),
                },
                "chunks": chunks,
                "chunk_links": chunk_links,
                "embedding_dimensions": len(embeddings[0]) if embeddings else 1536,
            },
        )

        return {
            "extracted_triples": triples,
            "graph_report": __import__("json").dumps(
                {"triples": upsert_result, "wikidata": wk_bundle, "chunks": chunks_result}
            ),
            "plain_text_for_extraction": user_prompt,
            "wikidata_enrichment": wk_bundle,
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "chunks": [{"chunk_id": c["chunk_id"], "start": c["start"], "end": c["end"], "text": c["text"][:200]} for c in chunks],
        }

    # Query mode: rely on Neo4j MCP search
    # 1) Lexical retrieval: vector search chunks by semantic similarity
    q_embedding = (await _embed_texts_openai([f"{company_name}\n{question}"]))[0]
    vector_hits = await call_mcp_tool(
        "graph_vector_search_chunks",
        {"embedding": q_embedding, "top_k": 8},
    )
    hit_chunks = vector_hits.get("results") or []
    hit_chunk_ids = [c.get("chunk_id") for c in hit_chunks if c.get("chunk_id")]

    # 2) Expand via entity graph: chunks -> entities -> neighbors -> related chunks
    expansion = await call_mcp_tool(
        "graph_expand_from_chunks",
        {"chunk_ids": hit_chunk_ids, "hops": 1, "per_entity_chunks": 3},
    )
    related_chunks = expansion.get("chunks") or []
    entities = expansion.get("entities") or []

    # 3) Domain traversal mode (existing behavior)
    system_prompt = (
        "You are a query planner for a Neo4j knowledge graph. "
        "Given a natural language question about a company, decide which strategy to use: "
        '"shortest_path" or "shared_board_members". Return JSON with keys '
        '"strategy", "source_company", and "target_company" if applicable.'
    )
    user_prompt = f"Company: {company_name}\nQuestion: {question}"

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    try:
        plan = __import__("json").loads(content)
    except Exception:
        plan = {}

    strategy = plan.get("strategy", "shortest_path")
    source_company = plan.get("source_company", company_name)
    target_company = plan.get("target_company", company_name)

    graph_search_result = await call_mcp_tool(
        "graph_search_paths",
        {
            "source_company": source_company,
            "target_company": target_company,
            "strategy": strategy,
        },
    )

    return {
        "graph_results": graph_search_result,
        "retrieved_chunks": {
            "vector_hits": hit_chunks,
            "related_chunks": related_chunks,
        },
        "retrieved_entities": entities,
    }

