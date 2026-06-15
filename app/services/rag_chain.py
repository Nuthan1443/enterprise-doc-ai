import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from app.config import get_settings
from app.services.retriever import RetrieverService

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Prompt Template ───────────────────────────────────────────────────────────
# This is the most important prompt engineering decision in the pipeline.
# "Answer ONLY from the context" prevents hallucination.
# "If the answer is not in the context" gives the LLM a graceful fallback.
# "Cite the source" forces grounded responses with document attribution.

RAG_PROMPT_TEMPLATE = """You are an enterprise document assistant. \
Answer the user's question using ONLY the information provided in the context below.

If the answer is not contained in the context, respond with:
"I don't have enough information in the provided documents to answer this question."

Always cite the source document at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """
    Formats retrieved documents into a single context string.
    Includes source metadata so the LLM can cite the document.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted.append(
            f"[Source {i}: {source}, chunk {chunk_idx}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


class RAGChain:
    """
    LangChain LCEL RAG chain.
    Wires retriever → prompt → LLM → output parser into
    a single composable, async-compatible chain.
    """

    def __init__(self, retriever_service: RetrieverService | None = None):
        if retriever_service is None:
            retriever_service = RetrieverService()

        self.retriever_service = retriever_service

        # LLM — Groq is used for speed (low latency inference)
        # temperature=0 for factual RAG — we want deterministic answers
        # not creative generation
        self.llm = ChatGroq(
            model=settings.groq_model_name,
            temperature=0,
            api_key=settings.groq_api_key,
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()

        # Build the LCEL chain
        self.chain = self._build_chain()
        logger.info("RAGChain initialized")

    def _build_chain(self):
        """
        Builds the LCEL chain using pipe operator.

        RunnableParallel runs two things simultaneously:
        - "context": retrieves relevant chunks and formats them
        - "question": passes the query through unchanged

        Both outputs feed into the prompt template as {context} and {question}.
        """
        retriever = self.retriever_service.retriever

        return (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def invoke(self, question: str) -> dict:
        """
        Synchronous invocation. Returns answer with sources.
        Use for testing and simple scripts.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"RAG query: '{question[:80]}'")

        # Get answer from chain
        answer = self.chain.invoke(question)

        # Get sources separately for structured response
        docs = self.retriever_service.retrieve(question)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(docs),
        }

    async def ainvoke(self, question: str) -> dict:
        """
        Async invocation for FastAPI endpoints.
        Same logic as invoke() but non-blocking.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Async RAG query: '{question[:80]}'")

        answer = await self.chain.ainvoke(question)
        docs = self.retriever_service.retrieve(question)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(docs),
        }