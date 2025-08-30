from langchain_community.retrievers import WikipediaRetriever
retriever=WikipediaRetriever(top_k_results=2,lang="en")
qeury="Can you tell me about elon musk?"
result=retriever.invoke(qeury)
for i,docs in enumerate(result):
    print(i+1)
    print(docs.page_content)