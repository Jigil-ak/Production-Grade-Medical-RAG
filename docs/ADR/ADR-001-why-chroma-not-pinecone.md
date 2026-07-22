# ADR-001: ChromaDB over Pinecone for Vector Storage

## Status
Accepted

## Context

The previous iteration of this project (`Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS`) used Pinecone as a managed cloud vector database. While Pinecone provides a robust, scalable vector search service, it introduces several issues for this rebuild:

1. **Network dependency**: Every retrieval query requires an API call to Pinecone's cloud, adding latency and creating a single point of failure for a system that should work offline during development and testing.
2. **Free-tier limitations**: Pinecone's free tier caps at 1 index with limited dimensions and storage. Our ~1,500-chunk corpus fits today, but the free tier's restrictions make iterative testing (clearing and re-indexing) cumbersome.
3. **Cost at scale**: Beyond the free tier, Pinecone charges per-read-unit and per-write-unit, which is misaligned with a system designed to run entirely on free-tier APIs.
4. **RAM constraint**: The project operates under a ~4GB RAM ceiling. A cloud vector DB adds the overhead of HTTP client connections and serialization/deserialization, while offering no local caching benefit.

## Decision

Use **ChromaDB in persistent-directory mode** (`chromadb.PersistentClient(path=...)`).

- Embedded: runs in-process, no separate server or daemon.
- Persistent: data survives process restarts via a local directory (`data/processed/chroma`).
- Zero network dependency for retrieval — queries are local disk reads.
- O(1) chunk lookups by using the `chunk_id` hash directly as the Chroma document ID.

## Consequences

### Benefits
- No cloud API dependency for retrieval — works fully offline.
- No API key management for vector storage.
- O(1) lookups by chunk_id (used directly as Chroma document ID).
- Simpler deployment: no Pinecone provisioning step.
- Idempotent ingestion: same chunk_id = same document, no duplicates.

### Tradeoffs
- **Single-node only**: ChromaDB embedded mode does not support distributed queries. Acceptable for a single-document medical Q&A system; would need revisiting for a multi-tenant or multi-document production deployment.
- **No managed backups**: Data lives on local disk. Mitigated by the fact that the entire index can be rebuilt from the source PDF in ~30 seconds.
- **No similarity search optimizations at scale**: ChromaDB's embedded HNSW index is fast for our ~1,500-chunk corpus but may degrade at 100k+ chunks. Not a concern for this project's scope.

## Template Note
This ADR format serves as the template for:
- **ADR-002**: Reranker model choice (TinyBERT vs MiniLM-L-6, Phase 2)
- **ADR-003**: Citation support threshold recalibration (Phase 3)
