from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = Path("llm-day3/data")
INDEX_DIR = Path("llm-day3/faiss_index")


def main():
    # 1. Load each text file.
    documents = []

    for path in DATA_DIR.glob("*.txt"):
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

        file_documents = loader.load()

        # Attach the filename as metadata.
        for document in file_documents:
            document.metadata["source"] = path.name

        documents.extend(file_documents)

    if not documents:
        raise ValueError("No .txt files found in the data directory.")

    print(f"Loaded {len(documents)} documents.")

    # 2. Split documents into smaller chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # 3. Load a local embedding model.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Embed the chunks and create a FAISS index.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # 5. Save the index locally.
    vector_store.save_local(str(INDEX_DIR))

    print(f"Saved index to: {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
