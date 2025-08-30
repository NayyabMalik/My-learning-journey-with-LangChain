from langchain_community.document_loaders import PyPDFLoader
pypdf_loader=PyPDFLoader(r"RAG\Document_Loaders\preprints202411.0566.v1.pdf")
docs=pypdf_loader.load()
print(len(docs))
print(docs[0].metadata)