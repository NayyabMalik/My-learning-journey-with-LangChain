from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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

vector_store=FAISS.from_documents(
    docs,
    embeddings
)



llm= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)
compressor=LLMChainExtractor.from_llm(
    llm
)
retrevier=vector_store.as_retriever(
        kwargs=
        {
            "k":2
        }
    )
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retrevier
)

query="who is famous person in cricket?"
result=compression_retriever.invoke(query)
print(result)