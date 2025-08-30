from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Load HuggingFace embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """
It’s a utility in LangChain that breaks large documents (text) into smaller, overlapping chunks.
This is super useful because LLMs can’t process huge texts at once, so you split them into manageable pieces before embedding, retrieval, or passing to the model.
"""

result = splitter.split_text(text)
print(result)
