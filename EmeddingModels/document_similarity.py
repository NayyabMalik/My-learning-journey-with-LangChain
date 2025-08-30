from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Specify the model name
model_name = "ibm-granite/granite-embedding-english-r2"
# Fallback model if the above fails: model_name = "sentence-transformers/all-MiniLM-L6-v2"

try:
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # List of documents and query
    documents = ["hi, i m ok", "i m doing my work", "i m outside of house"]
    query = "where are you?"

    # Embed the documents (use embed_documents for a list of texts)
    document_embeddings = embeddings.embed_documents(documents)

    # Embed the query (use embed_query for a single text)
    query_embedding = embeddings.embed_query(query)

    # Convert embeddings to NumPy arrays for cosine similarity
    document_embeddings = np.array(document_embeddings)
    query_embedding = np.array([query_embedding])  # Shape (1, n_features)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    # Find the index of the document with the maximum similarity
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]
    max_similarity_doc = documents[max_similarity_idx]

    # Print all similarities for reference
    print("Cosine similarities between query and documents:")
    for i, doc in enumerate(documents):
        print(f"Document: '{doc}' -> Similarity: {similarities[i]:.4f}")

    # Print the document with the maximum similarity
    print(f"\nDocument with maximum similarity: '{max_similarity_doc}' -> Similarity: {max_similarity:.4f}")

except Exception as e:
    print(f"An error occurred: {str(e)}")