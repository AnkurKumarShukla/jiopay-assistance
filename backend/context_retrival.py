import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv(os.path.join("backend", ".env"))

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Index Names
DENSE_INDEX = "rag-chatbot-dense"
SPARSE_INDEX = "rag-chatbot-sparse"

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing indexes
dense_idx = pc.Index(host="https://rag-chatbot-dense-rm8ktr2.svc.aped-4627-b74a.pinecone.io")
sparse_idx = pc.Index(host="https://rag-chatbot-sparse-rm8ktr2.svc.aped-4627-b74a.pinecone.io")

# Load OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_sparse_embedding(text):
    """
    Generate sparse embeddings using Pinecone's sparse embedding model.
    """
    response = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[text],
        parameters={"input_type": "query", "return_tokens": False}
    )
    return response.data[0]

def hybrid_search(query, top_k=5):
    """
    Perform hybrid search using both sparse and dense embeddings.
    """
    # Get dense and sparse embeddings
    dense_embedding = embedding_model.embed_query(query)
    sparse_embedding = get_sparse_embedding(query)
    
    # Query both indexes
    dense_results = dense_idx.query(
        namespace="chatbot-namespace", 
        vector=dense_embedding, 
        top_k=top_k, 
        include_metadata=True, 
        include_values=True
    )
    
    sparse_results = sparse_idx.query(
        sparse_vector={
            "indices": sparse_embedding["sparse_indices"],
            "values": sparse_embedding["sparse_values"]
        },
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )

    # Merge results
    combined_results = dense_results["matches"] + sparse_results["matches"]
    combined_results.sort(key=lambda x: x["score"], reverse=True)  # Sort by score

    return combined_results[:top_k]  # Return top results

# Example usage
if __name__ == "__main__":
    query = "What if a P2PM Merchant merchants breaches ₹ 1,00,000/- monthly limit?"
    results = hybrid_search(query)

    print("\n🔹 Extracted Context:")
    for idx, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source_text = metadata.get("source_text", "No context available")
        url = metadata.get("url", "No URL provided")
        section = metadata.get("section", "Unknown Section")
        
        print(f"{idx}. **Section:** {section} | **URL:** {url} | **Score:** {result['score']}")
        print(f"   📌 Context: {source_text}\n")
