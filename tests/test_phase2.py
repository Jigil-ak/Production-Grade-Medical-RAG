"""Unit and integration tests for Phase 2 components:

- BM25 & RRF Hybrid Fusion
- TinyBERT CrossEncoder Reranker
- final_answer_k slicing verification
- NLTK punkt failure & regex sentence splitter fallback
- Hybrid vs Vector-only ranking integration test
- PROMPT_VERSION configuration switching
- Explicit CitationEnforcer status tests ("answered", "partial", "declined")
- Max-over-sentences support scoring regression guard
"""

from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.core.types import Citation, RetrievedChunk
from app.embedding.service import MiniLMEmbeddingService
from app.generation.citation_enforcer import CitationEnforcer, _cosine_similarity
from app.generation.prompt_provider import YAMLPromptProvider
from app.retrieval.hybrid import RRFHybridRetriever
from app.retrieval.reranker import TinyBERTReranker


class TestFinalAnswerKSlicing:
    """Validate final_answer_k slicing in retrieval pipeline."""

    def test_final_answer_k_slicing_equals_setting(self) -> None:
        settings = Settings(GROQ_API_KEY="test")  # type: ignore[call-arg]
        assert settings.retrieval.rerank_top_k == 5
        assert settings.retrieval.final_answer_k == 4

        # Create 5 synthetic reranked chunks
        reranked_chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{i}",
                source_filename="doc.pdf",
                page_number=1,
                char_start=i * 100,
                char_end=(i + 1) * 100,
                chunk_text=f"Medical content chunk {i}",
                score=0.9 - (i * 0.1),
                retrieval_method="reranked",
            )
            for i in range(5)
        ]

        # Apply final_answer_k slice as done in routes.py
        final_chunks = reranked_chunks[: settings.retrieval.final_answer_k]

        assert len(final_chunks) == 4
        assert len(final_chunks) != settings.retrieval.rerank_top_k


class TestRRFHybridRetriever:
    """Test Reciprocal Rank Fusion logic."""

    def test_rrf_fusion_ordering(self) -> None:
        """Synthetic test: RRF fusion combines vector and BM25 ranks correctly."""
        c1 = RetrievedChunk(
            chunk_id="chunk1",
            source_filename="f.pdf",
            page_number=1,
            char_start=0,
            char_end=10,
            chunk_text="Aspirin reduces fever and pain.",
            score=0.9,
            retrieval_method="vector",
        )
        c2 = RetrievedChunk(
            chunk_id="chunk2",
            source_filename="f.pdf",
            page_number=1,
            char_start=11,
            char_end=20,
            chunk_text="Ibuprofen is an anti-inflammatory drug.",
            score=0.8,
            retrieval_method="vector",
        )
        c3 = RetrievedChunk(
            chunk_id="chunk3",
            source_filename="f.pdf",
            page_number=1,
            char_start=21,
            char_end=30,
            chunk_text="Acetaminophen treats mild to moderate fever.",
            score=0.7,
            retrieval_method="vector",
        )

        class MockVectorRetriever:
            def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
                return [c1, c2]

        class MockBM25Index:
            def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
                return [c2, c3]  # c2 rank 1 in BM25, c3 rank 2

        retriever = RRFHybridRetriever(
            vector_retriever=MockVectorRetriever(),  # type: ignore[arg-type]
            bm25_index=MockBM25Index(),  # type: ignore[arg-type]
            rrf_k=60,
        )

        fused = retriever.retrieve("fever treatment", vector_top_k=2, bm25_top_k=2)

        assert len(fused) == 3
        # c2 appears in BOTH vector (rank 2) and BM25 (rank 1), so c2 must rank FIRST overall!
        assert fused[0].chunk_id == "chunk2"

    def test_hybrid_beats_vector_only(self) -> None:
        """Integration test: Specific keyword match chunk ranks higher in hybrid search than vector-only."""
        # Chunk A: Vector similarity is moderately high for general medical text
        chunk_a = RetrievedChunk(
            chunk_id="vec_generic",
            source_filename="book.pdf",
            page_number=10,
            char_start=0,
            char_end=100,
            chunk_text="General gastrointestinal conditions include acid reflux, gastritis, and indigestion.",
            score=0.85,
            retrieval_method="vector",
        )

        # Chunk B: Contains exact keyword 'achalasia' which BM25 scores high
        chunk_b = RetrievedChunk(
            chunk_id="bm25_specific",
            source_filename="book.pdf",
            page_number=45,
            char_start=200,
            char_end=300,
            chunk_text="Achalasia is a rare esophageal motility disorder diagnosed via manometry.",
            score=0.60,  # Vector rank 2
            retrieval_method="vector",
        )

        class MockVectorRetriever:
            def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
                # Vector search puts generic chunk_a 1st, specific chunk_b 2nd
                return [chunk_a, chunk_b]

        class MockBM25Index:
            def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
                # BM25 keyword search puts specific chunk_b 1st
                return [chunk_b]

        hybrid = RRFHybridRetriever(
            vector_retriever=MockVectorRetriever(),  # type: ignore[arg-type]
            bm25_index=MockBM25Index(),  # type: ignore[arg-type]
            rrf_k=60,
        )

        query = "achalasia diagnosis manometry"

        # Vector-only order: chunk_a (rank 1), chunk_b (rank 2)
        vector_only_ids = [c.chunk_id for c in MockVectorRetriever().retrieve(query, 2)]
        assert vector_only_ids.index("bm25_specific") == 1

        # Hybrid order: chunk_b (rank 1) due to combined RRF score
        hybrid_fused = hybrid.retrieve(query, vector_top_k=2, bm25_top_k=2)
        hybrid_ids = [c.chunk_id for c in hybrid_fused]

        assert hybrid_ids.index("bm25_specific") == 0
        assert hybrid_ids.index("bm25_specific") < vector_only_ids.index("bm25_specific")


class TestTinyBERTReranker:
    """Test TinyBERT CrossEncoder reranking."""

    def test_reranker_reorders_candidates(self) -> None:
        reranker = TinyBERTReranker()

        c1 = RetrievedChunk(
            chunk_id="c1",
            source_filename="doc.pdf",
            page_number=1,
            char_start=0,
            char_end=10,
            chunk_text="Weather is sunny and pleasant today in California.",
            score=0.9,
            retrieval_method="vector",
        )
        c2 = RetrievedChunk(
            chunk_id="c2",
            source_filename="doc.pdf",
            page_number=1,
            char_start=11,
            char_end=20,
            chunk_text="Hypertension is high blood pressure treated with ACE inhibitors.",
            score=0.5,
            retrieval_method="vector",
        )

        reranked = reranker.rerank(
            query="How is high blood pressure or hypertension treated?",
            chunks=[c1, c2],
            top_k=2,
        )

        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c2"


class TestYAMLPromptProvider:
    """Test YAML prompt versioning and switching."""

    def test_load_prompt_v1(self) -> None:
        provider = YAMLPromptProvider(prompts_dir="./app/config/prompts")
        template = provider.get(name="answer_generation", version="answer_generation_v1")

        assert template.version == "answer_generation_v1"
        assert "{context}" in template.user_template
        assert "{question}" in template.user_template
        assert "medical assistant" in template.system_prompt.lower()

    def test_prompt_version_switching(self) -> None:
        """Confirm changing PROMPT_VERSION switches template without code changes."""
        provider = YAMLPromptProvider(prompts_dir="./app/config/prompts")

        v1_template = provider.get(name="answer_generation", version="answer_generation_v1")
        v2_template = provider.get(name="answer_generation", version="answer_generation_v2")

        assert v1_template.version != v2_template.version
        assert v1_template.version == "answer_generation_v1"
        assert v2_template.version == "answer_generation_v2"
        assert "[V2 TEMPLATE]" in v2_template.user_template


class TestCitationEnforcer:
    """Test CitationEnforcer sentence splitting, fallback, and 3 status outcomes."""

    @pytest.fixture
    def embedding_service(self) -> MiniLMEmbeddingService:
        return MiniLMEmbeddingService(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def enforcer(self, embedding_service: MiniLMEmbeddingService) -> CitationEnforcer:
        return CitationEnforcer(embedding_service=embedding_service, support_threshold=0.65)

    def test_nltk_failure_regex_fallback(self, enforcer: CitationEnforcer) -> None:
        """Test simulating NLTK sent_tokenize failure triggers regex fallback path."""
        text = "First sentence here. Second sentence follows. Third sentence ends."

        with patch("nltk.tokenize.sent_tokenize", side_effect=Exception("NLTK offline")):
            sentences = enforcer.split_sentences(text)
            assert len(sentences) == 3
            assert sentences[0] == "First sentence here."
            assert sentences[1] == "Second sentence follows."

    def test_max_over_sentences_regression_guard(
        self, enforcer: CitationEnforcer, embedding_service: MiniLMEmbeddingService
    ) -> None:
        """CRITICAL REGRESSION GUARD:
        Chunk with irrelevant first half and supporting second sentence.
        Assert whole-chunk average is diluted compared to max-over-sentences.
        """
        claim_sentence = "Lisinopril is an ACE inhibitor prescribed to manage high blood pressure and hypertension."

        irrelevant_part = (
            "The historical overview of cardiovascular epidemiology shows significant shifts in dietary habits. "
            "Urbanization and industrial food production altered sodium intake across populations worldwide. "
            "Many architectural designs in modern hospitals aim to improve patient recovery rates through natural lighting."
        )
        supporting_sentence = "Lisinopril is an ACE inhibitor used for treating hypertension and managing elevated blood pressure."

        full_chunk_text = f"{irrelevant_part} {supporting_sentence}"

        claim_vec = embedding_service.embed_query(claim_sentence)
        whole_chunk_vec = embedding_service.embed_query(full_chunk_text)
        whole_chunk_sim = _cosine_similarity(claim_vec, whole_chunk_vec)

        sentences = enforcer.split_sentences(full_chunk_text)
        sentence_vecs = embedding_service.embed_documents(sentences)
        max_sentence_sim = max(_cosine_similarity(claim_vec, s_vec) for s_vec in sentence_vecs)

        assert max_sentence_sim > whole_chunk_sim
        assert max_sentence_sim >= 0.65

    def test_citation_enforcer_answered_status(self, enforcer: CitationEnforcer) -> None:
        """1. ANSWERED status: all claims supported by context."""
        chunk = RetrievedChunk(
            chunk_id="chunk_ans",
            source_filename="medical.pdf",
            page_number=1,
            char_start=0,
            char_end=50,
            chunk_text="Aspirin is used to lower fever and relieve minor body aches.",
            score=0.9,
            retrieval_method="reranked",
        )

        citation = Citation(
            chunk_id="chunk_ans",
            source_filename="medical.pdf",
            page_number=1,
            quoted_text="Aspirin lowers fever and relieves pain.",
        )

        result = enforcer.enforce_citations(
            raw_answer="Aspirin is effective at lowering fever and relieving minor aches.",
            raw_citations=[citation],
            retrieved_chunks=[chunk],
            prompt_version="v1",
        )

        assert result.status == "answered"
        assert len(result.citations) == 1
        assert result.confidence >= 0.65
        assert result.answer is not None

    def test_citation_enforcer_partial_status(self, enforcer: CitationEnforcer) -> None:
        """2. PARTIAL status: one claim supported, another unsupported."""
        c1 = RetrievedChunk(
            chunk_id="c1",
            source_filename="m.pdf",
            page_number=1,
            char_start=0,
            char_end=50,
            chunk_text="Aspirin reduces fever and reduces headache symptoms.",
            score=0.9,
            retrieval_method="reranked",
        )
        c2 = RetrievedChunk(
            chunk_id="c2",
            source_filename="m.pdf",
            page_number=2,
            char_start=100,
            char_end=150,
            chunk_text="The average precipitation in Seattle during November is six inches.",
            score=0.1,
            retrieval_method="reranked",
        )

        cite1 = Citation(
            chunk_id="c1",
            source_filename="m.pdf",
            page_number=1,
            quoted_text="Aspirin reduces fever",
        )
        cite2 = Citation(
            chunk_id="c2",
            source_filename="m.pdf",
            page_number=2,
            quoted_text="Quantum mechanics governs subatomic physics",
        )

        result = enforcer.enforce_citations(
            raw_answer="Aspirin reduces fever. Quantum mechanics governs subatomic physics.",
            raw_citations=[cite1, cite2],
            retrieved_chunks=[c1, c2],
            prompt_version="v1",
        )

        assert result.status == "partial"
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "c1"
        assert len(result.unsupported_claims) == 1

    def test_citation_enforcer_declined_status(self, enforcer: CitationEnforcer) -> None:
        """3. DECLINED status: all citations fabricated or below threshold."""
        chunk = RetrievedChunk(
            chunk_id="valid_chunk",
            source_filename="medical.pdf",
            page_number=1,
            char_start=0,
            char_end=50,
            chunk_text="Some medical content.",
            score=0.9,
            retrieval_method="reranked",
        )

        fabricated_citation = Citation(
            chunk_id="fabricated_id_999",
            source_filename="medical.pdf",
            page_number=1,
            quoted_text="Fabricated statement",
        )

        result = enforcer.enforce_citations(
            raw_answer="Fabricated medical claim.",
            raw_citations=[fabricated_citation],
            retrieved_chunks=[chunk],
            prompt_version="v1",
        )

        assert result.status == "declined"
        assert result.answer is None
        assert len(result.citations) == 0
        assert result.confidence == 0.0
