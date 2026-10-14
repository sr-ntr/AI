from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "llm-day3/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

question = "What happens if a customer is charged twice?"

documents = retriever.invoke(question)

for i, document in enumerate(documents, start=1):
    print(f"\n--- Retrieved document {i} ---")
    print("Source:", document.metadata.get("source"))
    print(document.page_content)
