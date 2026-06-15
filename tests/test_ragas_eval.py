import pytest
from app.services.rag_chain import RAGChain
from app.services.retriever import RetrieverService
from app.evaluation.ragas_eval import run_ragas_evaluation
from data.golden_qa import GOLDEN_QA


@pytest.fixture(scope="module")
def rag_chain():
    return RAGChain()


@pytest.fixture(scope="module")
def retriever():
    return RetrieverService()


def test_ragas_evaluation(rag_chain, retriever):
    test_cases = []

    for qa in GOLDEN_QA:
        result = rag_chain.invoke(qa["question"])
        docs = retriever.retrieve(qa["question"])
        contexts = [doc.page_content for doc in docs]

        test_cases.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": qa["ground_truth"],
        })

        print(f"\n[QA] Q: {qa['question']}")
        print(f"     A: {result['answer'][:100]}...")

    scores = run_ragas_evaluation(test_cases)

    print(f"\n{'='*50}")
    print(f"RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    for metric, score in scores.items():
        print(f"  {metric:25} {score:.4f}")
    print(f"{'='*50}")

    assert scores.get("faithfulness", 0) >= 0.5, \
        f"Faithfulness too low: {scores.get('faithfulness')}"
    assert scores.get("answer_relevancy", 0) >= 0.5, \
        f"Answer relevancy too low: {scores.get('answer_relevancy')}"
    
    print(f"\n{'='*50}")
    print(f"RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:25} {score:.4f}  {bar}")
    print(f"{'='*50}")