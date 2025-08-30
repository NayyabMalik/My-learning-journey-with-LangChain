from langchain_community.document_loaders import CSVLoader
csv_Loader=CSVLoader(r"RAG\Document_Loaders\Copy of Osama sales.csv")
docs=csv_Loader.load()
print(len(docs))
print(docs[0].metadata)