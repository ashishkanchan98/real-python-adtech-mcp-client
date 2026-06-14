import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


async def create_mcp_tools(mcp_server_url: str) -> list:
    """
    Connect to the real MCP server via SSE transport and return LangChain tools.
    Uses official MCP protocol (JSON-RPC 2.0 over SSE) instead of custom HTTP REST.
    The MultiServerMCPClient stays open — caller must manage its lifecycle.
    """
    sse_url = mcp_server_url.rstrip("/") + "/sse"
    logger.info("Connecting to MCP server via SSE: %s", sse_url)

    client = MultiServerMCPClient({
        "adtech": {
            "url": sse_url,
            "transport": "sse",
        }
    })
    await client.__aenter__()
    tools = client.get_tools()
    logger.info("Loaded %d tools from MCP server via SSE", len(tools))
    return tools, client


# ── Knowledge Base ────────────────────────────────────────────────────────────
# Native LangChain tool — AI capability lives here, not on the MCP server.
# KB_PROVIDER=local  → ChromaDB (embedded, seeded from kb-docs/, no GCP needed)
# KB_PROVIDER=gcp    → Vertex AI Search (Discovery Engine, requires GCP_PROJECT_ID)

# Singleton ChromaDB collection — initialised once per process
_chroma_collection = None


def _get_chroma_collection(settings):
    """Lazy-init ChromaDB collection. Seeds from kb-docs/ on first call."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    # all-MiniLM-L6-v2 is ~90 MB, downloaded once to ~/.cache/huggingface/
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=settings.chroma_db_path)

    collection = client.get_or_create_collection("adtech_kb", embedding_function=ef)

    if collection.count() == 0:
        _seed_chroma(collection, settings.kb_docs_dir)

    logger.info("ChromaDB ready  docs=%d  path=%s", collection.count(), settings.chroma_db_path)
    _chroma_collection = collection
    return collection


def _seed_chroma(collection, kb_docs_dir: str) -> None:
    """Load all kb-docs/*.json into the ChromaDB collection."""
    docs_path = Path(kb_docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"kb-docs not found at '{kb_docs_dir}'. "
            "Run from the python-adtech-mcp-client directory or set KB_DOCS_DIR."
        )

    json_files = sorted(docs_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {kb_docs_dir}")

    ids, documents, metadatas = [], [], []
    for f in json_files:
        with open(f) as fp:
            doc = json.load(fp)
        struct = doc["structData"]
        title = struct.get("title", "")
        content = struct.get("content", "")
        ids.append(doc["id"])
        documents.append(f"{title}\n\n{content}")
        metadatas.append({
            "title": title,
            "category": struct.get("category", ""),
            "link": struct.get("link", ""),
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("ChromaDB seeded with %d documents from %s", len(ids), kb_docs_dir)


def _do_kb_search_local(query: str, max_results: int, settings) -> dict:
    """Search using local ChromaDB (no GCP required)."""
    try:
        collection = _get_chroma_collection(settings)
        n = min(max_results, collection.count())
        results = collection.query(query_texts=[query], n_results=n)

        docs = []
        for doc_id, meta, distance in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs.append({
                "id": doc_id,
                "title": meta.get("title", ""),
                "snippet": meta.get("category", ""),
                "link": meta.get("link", ""),
                "score": round(1 - distance, 3),
            })

        logger.info("ChromaDB search returned %d results for query='%s'", len(docs), query)
        return {"query": query, "count": len(docs), "summary": "", "results": docs, "source": "local"}

    except Exception as exc:
        logger.error("ChromaDB search failed: %s", exc)
        return {"error": str(exc), "query": query, "count": 0, "results": []}


def _do_kb_search_gcp(query: str, max_results: int, settings) -> dict:
    """Search using Vertex AI Search (Discovery Engine) on GCP."""
    if not settings.gcp_project_id or not settings.kb_datastore_id:
        logger.error("KB_PROVIDER=gcp but GCP_PROJECT_ID is not set")
        return {"error": "GCP_PROJECT_ID is required when KB_PROVIDER=gcp", "query": query, "count": 0, "results": []}
    try:
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.SearchServiceClient()
        serving_config_name = client.serving_config_path(
            project=settings.gcp_project_id,
            location="global",
            data_store=settings.kb_datastore_id,
            serving_config=settings.kb_serving_config,
        )
        request = discoveryengine.SearchRequest(
            serving_config=serving_config_name,
            query=query,
            page_size=max_results,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True, max_snippet_count=2
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    include_citations=True,
                    summary_result_count=max_results,
                    ignore_adversarial_query=True,
                ),
            ),
        )
        response = client.search(request)

        results = []
        for r in response.results:
            derived = r.document.derived_struct_data

            def _field(name: str) -> str:
                val = derived.get(name)
                if val is None:
                    return ""
                if hasattr(val, "list_value") and val.list_value.values:
                    parts = []
                    for v in val.list_value.values:
                        if hasattr(v, "struct_value"):
                            s = v.struct_value.get("snippet")
                            parts.append(s.string_value if s else "")
                        else:
                            parts.append(v.string_value)
                    return " ".join(p for p in parts if p)
                return val.string_value if hasattr(val, "string_value") else str(val)

            results.append({
                "id": r.id,
                "title": _field("title"),
                "snippet": _field("snippets"),
                "link": _field("link"),
            })

        summary = ""
        if hasattr(response, "summary") and response.summary:
            summary = response.summary.summary_text

        logger.info("Vertex AI Search returned %d results for query='%s'", len(results), query)
        return {"query": query, "count": len(results), "summary": summary, "results": results, "source": "gcp"}

    except Exception as exc:
        logger.error("Vertex AI Search failed: %s", exc)
        return {"error": str(exc), "query": query, "count": 0, "results": []}


def _do_kb_search(query: str, max_results: int, settings) -> dict:
    """Dispatch to local ChromaDB or GCP Vertex AI Search based on KB_PROVIDER."""
    if settings.kb_provider == "gcp":
        return _do_kb_search_gcp(query, max_results, settings)
    return _do_kb_search_local(query, max_results, settings)


def create_kb_tool(settings) -> StructuredTool:
    """
    Native LangChain tool for Knowledge Base search.
    KB_PROVIDER=local  → ChromaDB (embedded, kb-docs/ seeded on first call)
    KB_PROVIDER=gcp    → Vertex AI Search (Discovery Engine)
    """
    class KbSearchInput(BaseModel):
        query: str = Field(description="Natural language search query")
        max_results: int = Field(default=3, description="Max docs to return (default 3)")

    def _sync(query: str, max_results: int = 3) -> str:
        return json.dumps(_do_kb_search(query, max_results, settings))

    async def _async(query: str, max_results: int = 3) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: json.dumps(_do_kb_search(query, max_results, settings))
        )

    _sync.__name__ = "searchKnowledgeBase"
    _async.__name__ = "searchKnowledgeBase"

    return StructuredTool(
        name="searchKnowledgeBase",
        description="Searches the AdTech support knowledge base for troubleshooting guides, policy docs, and how-to articles.",
        args_schema=KbSearchInput,
        func=_sync,
        coroutine=_async,
    )
