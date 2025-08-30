from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "ibm-granite/granite-embedding-english-r2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

text="hi,how are you"
embedding = embeddings.embed_query(text)
print(str(embedding))