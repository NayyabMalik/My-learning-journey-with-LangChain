from langchain.document_loaders import TextLoader

text_loader = TextLoader(
    r"RAG\Document_Loaders\transcription.txt",
    encoding="latin-1" 
)
docs = text_loader.load()
print(docs)
