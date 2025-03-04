import os
import json
import nltk
import re
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, VectorType
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from nltk.corpus import stopwords

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

# === Create Dense Index ===
if DENSE_INDEX not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=DENSE_INDEX,
        dimension=1536,  # OpenAI embedding dimension
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        vector_type=VectorType.DENSE
    )
dense_idx = pc.Index(name=DENSE_INDEX)

# === Create Sparse Index ===
if SPARSE_INDEX not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=SPARSE_INDEX,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        vector_type=VectorType.SPARSE
    )
sparse_idx = pc.Index(name=SPARSE_INDEX)

# Load OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Download stopwords for category extraction
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def extract_category(text, max_words=3):
    """
    Extracts category from text by removing stopwords and keeping key words.
    """
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize words
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_words[:max_words])  # Limit category size

# === Load PDFs & JSON Data ===
def load_pdfs(pdf_files):
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        for page in pages:
            documents.append(Document(page_content=page.page_content, metadata={"source": pdf}))
    return documents

def load_json(json_files):
    documents = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if "content" in data:
            url = data.get("url", "Unknown URL")
            for section in data["content"]:
                section_title = section.get("section", "Unknown Section")
                for item in section.get("text", []):
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    if question or answer:
                        doc = Document(
                            page_content=f"Q: {question}\nA: {answer}",
                            metadata={"section": section_title, "url": url}
                        )
                        documents.append(doc)

        elif "pages" in data:
            for page in data["pages"]:
                url = page.get("url", "Unknown URL")
                for item in page.get("content", []):
                    section = item.get("section", "Unknown Section")
                    text = item.get("text", "")
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={'section': section, 'url': url}
                        )
                        documents.append(doc)
    return documents

# === Chunk Documents ===
def chunk_documents(docs, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def prepare_dense_vectors(docs):
    vectors = []
    for i, doc in enumerate(docs):
        dense_embedding = embedding_model.embed_query(doc.page_content)
        category = extract_category(doc.page_content)

        vectors.append({
            "id": f"doc-{i}",
            "values": dense_embedding,
            "metadata": {
                "category": category,
                "source_text": doc.page_content,
                "origin": json.dumps(doc.metadata)  # Convert to string
            }
        })  
    return vectors




# === Prepare Sparse Vectors ===
def get_sparse_embedding(texts, batch_size=96):
    """
    Generate sparse embeddings using Pinecone's sparse model.
    """
    sparse_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        response = pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=batch_texts,
            parameters={"input_type": "passage", "return_tokens": False}
        )
        sparse_embeddings.extend(response.data)
    
    return sparse_embeddings

def prepare_sparse_vectors(docs):
    vectors = []
    texts = [doc.page_content for doc in docs]
    
    sparse_embeddings = get_sparse_embedding(texts)

    for i, doc in enumerate(docs):
        sparse_data = sparse_embeddings[i]

        if not sparse_data["sparse_indices"] or not sparse_data["sparse_values"]:
            print(f"Skipping document {i}: Sparse vector is empty.")
            continue

        category = extract_category(doc.page_content)

        vectors.append({
            "id": f"sparse-doc-{i}",
            "sparse_values": {
                "indices": sparse_data["sparse_indices"],
                "values": sparse_data["sparse_values"]
            },
            "metadata": {
                "category": category,
                "source_text": doc.page_content,
                "origin": json.dumps(doc.metadata)  # Convert to string
            }
        })
    return vectors

# === Ingest Data ===
def ingest_dense_vectors(docs, batch_size=50):
    vectors = prepare_dense_vectors(docs)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        dense_idx.upsert(vectors=batch, namespace="chatbot-namespace")
        print(f"Ingested batch {i // batch_size + 1} ({len(batch)} vectors) into Dense Index.")

def ingest_sparse_vectors(docs, batch_size=50):
    vectors = prepare_sparse_vectors(docs)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        sparse_idx.upsert(vectors=batch, namespace="chatbot-namespace")
        print(f"Ingested batch {i // batch_size + 1} ({len(batch)} vectors) into Sparse Index.")

# === Main Execution ===
if __name__ == "__main__":
    nltk.download("punkt")

    # Load and chunk data
    raw_docs = load_pdfs(["data_collection/Grievance-Redressal-Policy.pdf","data_collection\Policy-for-Selection-of-Directors-and-determining-Directors-Independence.pdf","data_collection\Remuneration-Policy-for-Directors-Key-Managerial-Personnel-and-other-Employees.pdf","data_collection\JPSL-Annual-Return-2023-24.pdf"]) + load_json([
        "data_collection/helpcentre.json",
        "data_collection/manual_data_extract.json"
    ])
    chunked_docs = chunk_documents(raw_docs)

    # Ingest Data into Pinecone
    ingest_dense_vectors(chunked_docs)
    ingest_sparse_vectors(chunked_docs)
