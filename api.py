import traceback

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import build_app
from state import LeadState


app = FastAPI(title="Lead Intelligence API")
compiled_app = build_app()


@app.get("/")
def root():
    return {"service": "Lead Intelligence API", "docs": "/docs", "ingest": "POST /leads/ingest", "query": "POST /leads/query"}


class IngestRequest(BaseModel):
    company_name: str
    domain: str


class QueryRequest(BaseModel):
    company_name: str
    domain: str
    question: str


async def _do_ingest(req: IngestRequest):
    state: LeadState = {
        "company_name": req.company_name,
        "domain": req.domain,
        "goal_type": "ingestion",
    }
    try:
        result = await compiled_app.ainvoke(state)
        raw = result.get("raw_web_content") or ""
        graph_report = result.get("graph_report") or "{}"
        try:
            report = __import__("json").loads(graph_report)
            skipped = report.get("skipped_extraction", False)
        except Exception:
            skipped = False
        return {
            "status": "ok",
            "content_length": len(raw),
            "skipped_extraction": skipped,
            "cleaned_content": result.get("plain_text_for_extraction") or "",
            "crawl_error": result.get("crawl_error") or None,
            "state": result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/leads/ingest")
async def ingest_lead(req: IngestRequest):
    return await _do_ingest(req)


@app.post("/leads/ingest/")
async def ingest_lead_trailing(req: IngestRequest):
    return await _do_ingest(req)


async def _do_query(req: QueryRequest):
    state: LeadState = {
        "company_name": req.company_name,
        "domain": req.domain,
        "question": req.question,
        "goal_type": "query",
    }
    try:
        result = await compiled_app.ainvoke(state)
        return {
            "status": "ok",
            "retrieved_chunks": result.get("retrieved_chunks") or {},
            "retrieved_entities": result.get("retrieved_entities") or [],
            "state": result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/leads/query")
async def query_lead(req: QueryRequest):
    return await _do_query(req)


@app.post("/leads/query/")
async def query_lead_trailing(req: QueryRequest):
    return await _do_query(req)

