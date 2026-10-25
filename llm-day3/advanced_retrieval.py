from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from sentence_transformers import CrossEncoder

load_dotenv()

DATA_DIR = Path("llm-day3/data")
INDEX_DIR = "llm-day3/faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's question as a concise search query.
        Preserve the user's intent.
        Do not assume facts that the user has not stated.
        Return only the rewritten query.""",
        ),
        ("human", "{question}"),
    ]
)

model = ChatGoogleGenerativeAI(
    model=os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    ),
    google_api_key=os.environ["GEMINI_API_KEY"],
)


def rewrite_query(model, question: str) -> str:
    messages = rewrite_prompt.invoke({"question": question})
    response = model.invoke(messages)
    rewritten = response.text.strip()
    return rewritten or question


def load_chunks():
    """Load and split source documents for keyword retrieval."""
    documents = []

    for path in DATA_DIR.glob("*.txt"):
        loaded = TextLoader(str(path), encoding="utf-8").load()

        for document in loaded:
            document.metadata["source"] = path.name

        documents.extend(loaded)

    if not documents:
        raise ValueError("No .txt files found in data/.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    return splitter.split_documents(documents)


def reciprocal_rank_fusion(*ranked_lists, c=60):
    """Combine ranked result lists using reciprocal rank fusion."""
    scores = {}
    documents_by_key = {}

    for ranked_documents in ranked_lists:
        for rank, document in enumerate(ranked_documents, start=1):
            # Include source as well as text to distinguish identical chunks
            # originating from different files.
            key = (
                document.metadata.get("source", "unknown"),
                document.page_content,
            )

            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
            documents_by_key[key] = document

    ranked_keys = sorted(scores, key=scores.get, reverse=True)

    return [documents_by_key[key] for key in ranked_keys]


def main():
    # Load the existing vector index.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Build a BM25 index from the current source documents.
    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    query = "What happens if a customer is charged twice?"

    # Query transformation
    original_query = "yo how the heck did you take my money again? i want it back"
    search_query = rewrite_query(model, original_query)

    print("Original:", original_query)
    print("Search query:", search_query)

    # Compare the results before deciding which query to use.
    original_results = vector_store.similarity_search(original_query, k=5)
    rewritten_results = vector_store.similarity_search(search_query, k=5)

    # Retrieve candidates independently.
    dense_results = vector_store.similarity_search(query, k=10)
    keyword_results = bm25_retriever.invoke(search_query)

    # Try using the three results (original, rewritten, dense)

    print("\n--- Dense retrieval ---")
    for document in rewritten_results:
        print(document.metadata.get("source"), ":", document.page_content)

    print("\n--- Keyword retrieval (BM25) ---")
    for document in keyword_results:
        print(document.metadata.get("source"), ":", document.page_content)

    # Merge rankings rather than adding incomparable raw scores.
    fused_results = reciprocal_rank_fusion(
        rewritten_results,
        keyword_results,
    )[:10]

    print("\n--- Hybrid retrieval (RRF) ---")
    for rank, document in enumerate(fused_results, start=1):
        print(f"\nRank {rank} | {document.metadata.get('source')}")
        print(document.page_content)

    # Rerank only the fused candidate set.
    reranker = CrossEncoder(RERANKER_MODEL)

    pairs = [(search_query, document.page_content) for document in fused_results]

    rerank_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(rerank_scores, fused_results),
        key=lambda item: item[0],
        reverse=True,
    )

    print("\n--- Reranked results ---")
    for rank, (score, document) in enumerate(reranked_results, start=1):
        print(
            f"\nRank {rank} | Score {float(score):.4f} "
            f"| Source: {document.metadata.get('source')}"
        )
        print(document.page_content)


if __name__ == "__main__":
    main()
