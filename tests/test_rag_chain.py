import pytest
from app.services.rag_chain import RAGChain


@pytest.fixture(scope="module")
def rag_chain():
    """Single RAGChain instance shared across all tests in this module."""
    return RAGChain()


def test_rag_chain_answers_leave_question(rag_chain):
    result = rag_chain.invoke("How many days of annual leave do employees get?")

    print(f"\n[RAG] Question: {result['question']}")
    print(f"[RAG] Answer: {result['answer']}")
    print(f"[RAG] Sources: {result['sources']}")
    print(f"[RAG] Chunks used: {result['chunks_used']}")

    assert result["answer"]
    assert "18" in result["answer"] or "leave" in result["answer"].lower()
    assert len(result["sources"]) > 0


def test_rag_chain_answers_remote_work_question(rag_chain):
    result = rag_chain.invoke("What is the remote work policy?")

    print(f"\n[RAG] Answer: {result['answer']}")
    assert result["answer"]
    assert "remote" in result["answer"].lower() or "days" in result["answer"].lower()


def test_rag_chain_handles_out_of_context_question(rag_chain):
    """
    Verifies the chain doesn't hallucinate when the answer
    isn't in the retrieved context.
    """
    result = rag_chain.invoke("What is the company's stock price?")

    print(f"\n[RAG] Out-of-context answer: {result['answer']}")
    # Should admit it doesn't know, not hallucinate
    answer_lower = result["answer"].lower()
    assert any(phrase in answer_lower for phrase in [
        "don't have", "not contained", "not in", "cannot", "no information"
    ])