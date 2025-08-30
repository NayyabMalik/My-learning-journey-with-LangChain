from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Use a stable embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Documents
docs = [
    Document(
        page_content="Babar Azam is one of the most stylish and consistent batsmen in modern cricket.",
        metadata={"team": "Karachi Kings"}
    ),
    Document(
        page_content="Shan Masood is a solid top-order batsman.",
        metadata={"team": "Multan Sultans"}
    ),
    Document(
        page_content="Shoaib Malik, a veteran all-rounder.",
        metadata={"team": "Peshawar Zalmi"}
    ),
    Document(
        page_content="Shaheen Afridi is one of the premier fast bowlers.",
        metadata={"team": "Lahore Qalandars"}
    ),
    Document(
        page_content="Fakhar Zaman is a destructive opening batsman.",
        metadata={"team": "Islamabad United"}
    )
]

vector_Store=Chroma(
    embedding_function=embedding,
    persist_directory="sample",
    collection_name="my_retreiver_db"
)

vector_Store.add_documents(docs)
retreiver=vector_Store.as_retriever(kwargs={"k":2})
query="who is best batman?"
result=retreiver.invoke(query)

print(result)
