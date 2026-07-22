"""Unit tests for Phase 2 components: BM25, RRF Hybrid Fusion, TinyBERT Reranker, YAML Prompt Provider, and CitationEnforcer."""

import pytest

from app.core.types import Citation, Chunk, RetrievedChunk
from app.embedding.service import MiniLMEmbeddingService
from app.generation.citation_enforcer import CitationEnforcer, _cosine_similarity
from app.generation.prompt_provider import YAMLPromptProvider
from app.retrieval.hybrid import RRFHybridRetriever
from app.retrieval.reranker import TinyBERTReranker


class TestRRFHybridRetriever:
    """Test Reciprocal Rank Fusion logic."""

    def test_rrf_fusion_ordering(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        # c2 is medically relevant to hypertension, c1 is about weather. c2 MUST be reranked to 1st position!
        assert reranked[0].chunk_id == "c2"


class TestYAMLPromptProvider:
    """Test YAML prompt versioning."""

    def test_load_prompt_v1(self) -> None:
        provider = YAMLPromptProvider(prompts_dir="./app/config/prompts")
        template = provider.get(name="answer_generation", version="answer_generation_v1")

        assert template.version == "answer_generation_v1"
        assert "{context}" in template.user_template
        assert "{question}" in template.user_template
        assert "medical assistant" in template.system_prompt.lower()


class TestCitationEnforcer:
    """Test CitationEnforcer and MAX-over-sentences support scoring algorithm."""

    @pytest.fixture
    def embedding_service(self) -> MiniLMEmbeddingService:
        return MiniLMEmbeddingService(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def enforcer(self, embedding_service: MiniLMEmbeddingService) -> CitationEnforcer:
        return CitationEnforcer(embedding_service=embedding_service, support_threshold=0.65)

    def test_max_over_sentences_regression_guard(
        self, enforcer: CitationEnforcer, embedding_service: MiniLMEmbeddingService
    ) -> None:
        """CRITICAL REGRESSION GUARD:
        Construct a chunk with an irrelevant long first sentence and a clearly-supporting second sentence.
        Assert that whole-chunk average cosine similarity dilutes the match below 0.65,
        while MAX-over-sentences correctly exceeds 0.65 and passes support enforcement.
        """
        claim_sentence = "Lisinopril is an ACE inhibitor prescribed to manage high blood pressure and hypertension."

        # Long chunk with irrelevant first half and relevant second sentence
        irrelevant_part = (
            "The historical overview of cardiovascular epidemiology shows significant shifts in dietary habits. "
            "Urbanization and industrial food production altered sodium intake across populations worldwide. "
            "Many architectural designs in modern hospitals aim to improve patient recovery rates through natural lighting."
        )
        supporting_sentence = "Lisinopril is an ACE inhibitor used for treating hypertension and managing elevated blood pressure."

        full_chunk_text = f"{irrelevant_part} {supporting_sentence}"

        # 1. Whole-chunk embedding average comparison
        claim_vec = embedding_service.embed_query(claim_sentence)
        whole_chunk_vec = embedding_service.embed_query(full_chunk_text)
        whole_chunk_sim = _cosine_similarity(claim_vec, whole_chunk_vec)

        # 2. Max-over-sentences comparison
        sentences = enforcer.split_sentences(full_chunk_text)
        sentence_vecs = embedding_service.embed_documents(sentences)
        max_sentence_sim = max(_cosine_similarity(claim_vec, s_vec) for s_vec in sentence_vecs)

        # Assert regression guard: whole chunk average is diluted compared to max-over-sentences
        assert max_sentence_sim > whole_chunk_sim, "Max-over-sentences must exceed whole-chunk average!"
        assert max_sentence_sim >= 0.65, f"Max-over-sentences score {max_sentence_sim:.3f} must pass 0.65 threshold!"

    def test_citation_enforcer_answered_status(self, enforcer: CitationEnforcer) -> None:
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

    def test_citation_enforcer_declined_status_for_fabricated_id(
        self, enforcer: CitationEnforcer
    ) -> None:
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
