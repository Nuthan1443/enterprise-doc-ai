import logging
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def build_ragas_llm():
    llm = ChatGroq(
        model=settings.groq_model_name,
        temperature=0,
        api_key=settings.groq_api_key,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation(test_cases: list[dict]) -> dict:
    logger.info(f"Running RAGAS evaluation on {len(test_cases)} test cases...")

    ragas_llm = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()

    samples = []
    for tc in test_cases:
        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=tc["answer"],
            retrieved_contexts=tc["contexts"],
            reference=tc["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)

    # Using ragas.metrics (not .collections) — required for non-OpenAI LLMs
    # .collections metrics only support InstructorLLM (OpenAI-only)
    # ragas.metrics accepts LangchainLLMWrapper with any provider
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Select only numeric columns — string columns cause TypeError on .mean()
    df = results.to_pandas()
    numeric_cols = df.select_dtypes(include="number").columns
    scores = df[numeric_cols].mean().to_dict()

    logger.info(f"RAGAS scores: {scores}")
    return scores