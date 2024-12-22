import os
from PyPDF2 import PdfReader
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import numpy as np


# Load and preprocess the PDF document
def load_pdf(file_path):
    """Load the PDF and extract its text."""
    reader = PdfReader(file_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


# Split the text into manageable chunks using RecursiveCharacterTextSplitter
def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split the text into smaller chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


# Generate embeddings for text chunks
def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for the given text chunks using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings


# Store embeddings and metadata in a FAISS vector database
def store_embeddings(embeddings, chunks, index_file):
    """Store embeddings in a FAISS index."""
    embeddings = np.array(embeddings)  # Ensure embeddings are numpy arrays
    dimension = embeddings.shape[1]   # Embedding vector size
    index = faiss.IndexFlatL2(dimension)  # L2 distance-based FAISS index
    index.add(embeddings)  # Add embeddings to the index
    faiss.write_index(index, index_file)  # Save the index to a file
    print(f"Index saved to {index_file}")
    return index


# Query the FAISS vector database for relevant chunks
def query_vector_db(query, index_file, model_name="all-MiniLM-L6-v2", k=3):
    """Query the FAISS index to retrieve the most relevant text chunks."""
    # Load FAISS index
    index = faiss.read_index(index_file)

    # Embed the query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # Perform similarity search
    distances, indices = index.search(query_embedding, k)
    return indices

# Generate a final answer using the retrieved chunks and LLM
def generate_answer(query, retrieved_chunks):
    """Generate an answer using LLM with the retrieved context."""
    llm=ChatGroq(model_name="Gemma2-9b-It")
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response




# Complete RAG pipeline for Claim Policy
def claim_pipeline(query):
    """Complete RAG pipeline for Claim Policy."""
    # Paths
    file_path = r"source\Claim_Policy.pdf"
    index_file = "embeddings/Claim_policy.index"

    # Step 1: Load and preprocess the document
    text = load_pdf(file_path)

    # Step 2: Split the text into chunks
    chunks = split_text(text)

    # Step 3: Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)

    # Step 4: Store the embeddings in a FAISS index
    store_embeddings(embeddings, chunks, index_file)

    # Step 5: Query the FAISS index
    indices = query_vector_db(query, index_file)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]

    # Step 6: Generate and return the answer
    answer = generate_answer(query, retrieved_chunks)
    return answer


# Test the Claim Policy pipeline
if __name__ == "__main__":
    # Example query
    user_query = "Health insurance covers medical expenses up to how much?"
    response = claim_pipeline(user_query)
    print("\n=== Answer ===\n")
    print(response.content)
    print("\n==============\n")
