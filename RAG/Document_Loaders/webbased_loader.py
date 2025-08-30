from langchain_community.document_loaders import WebBaseLoader
webbased_loader=WebBaseLoader("https://en.wikipedia.org/wiki/Black_hole")
docs=webbased_loader.load()
print(len(docs))