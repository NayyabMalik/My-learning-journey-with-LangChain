from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r"RAG\Document_Loaders\preprints202411.0566.v1.pdf")
docs=loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",   # how to split (can also use " " or "")
    chunk_size=100,    # max characters in each chunk
    chunk_overlap=10 # how much chunks overla
)

result=text_splitter.split_documents(docs)
print(result[0])
print(len(result))