# Medical RAG — Query Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Retriever
    participant VectorStore as ChromaDB
    participant BM25 as BM25 Index
    participant Reranker as TinyBERT Reranker
    participant LLM as Groq API
    participant Enforcer as Citation Enforcer

    Client->>API: POST /query {question}
    API->>Retriever: retrieve(question)

    Note over Retriever: Phase 1: vector only
    Note over Retriever: Phase 2: hybrid + rerank

    Retriever->>VectorStore: query(embedding, vector_top_k)
    VectorStore-->>Retriever: vector results

    Retriever->>BM25: search(question, bm25_top_k)
    BM25-->>Retriever: keyword results

    Note over Retriever: RRF fusion (k=60)

    Retriever->>Reranker: rerank(query, candidates)
    Reranker-->>Retriever: reranked top rerank_top_k

    Retriever-->>API: top chunks
    API->>LLM: generate(system_prompt, context + question)
    LLM-->>API: answer + cited chunk_ids

    API->>Enforcer: validate citations
    Note over Enforcer: 1. Verify chunk_ids in retrieved set
    Note over Enforcer: 2. Max-over-sentences support scoring
    Enforcer-->>API: {status, confidence, unsupported_claims}

    API-->>Client: {answer, citations, confidence, status}
```

## Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest PDFs from /data/raw |
| `/query` | POST | Ask a question, get answer + citations |
| `/chunk/{chunk_id}` | GET | O(1) lookup of a specific chunk |
| `/health` | GET | Health check |
