import os
import logging
from uuid import uuid4
from typing import List, Dict, Optional
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables from .env file (e.g., HF_TOKEN)
load_dotenv()

# --- Custom Embeddings Class using HuggingFace Inference API ---
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_token: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")  # Get token from args or env
        self.client = InferenceClient(model=self.model_name, token=self.hf_token)
        logging.info(f"Initialized HuggingFaceAPIEmbeddings with model: {self.model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"Embedding {len(texts)} documents")
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        logging.info("Embedding single query")
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Internal helper method to send texts to HuggingFace for embedding.
        """
        try:
            return self.client.feature_extraction(texts)
        except Exception as e:
            logging.error(f"HuggingFace API Error: {str(e)}")
            raise ValueError(f"HuggingFace API Error: {str(e)}")

# --- Vector Store Setup ---
embedding_model = HuggingFaceAPIEmbeddings()

# Splitter breaks text into manageable chunks for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Global FAISS vector store instance
vectorstore: Optional[FAISS] = None

# --- Function to Add Documents to Vector Index ---
def add_to_index(doc: Dict, doc_type: str, extra_metadata: Optional[Dict] = None):
    """
    Adds a parsed document to the FAISS index after chunking and embedding.
    
    Parameters:
    - doc: A dictionary with 'text' and optional 'metadata'
    - doc_type: A tag for identifying document category
    - extra_metadata: Any additional metadata to attach
    """
    global vectorstore
    logging.info(f"Adding document to index. Type: {doc_type}")
    
    # Split large text into smaller overlapping chunks
    chunks = splitter.create_documents([doc["text"]])
    texts = [chunk.page_content for chunk in chunks]
    logging.info(f"Split document into {len(texts)} chunks")

    # Attach metadata to each chunk
    documents = []
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "id": str(uuid4()),  # Unique ID per chunk
            "doc_type": doc_type,
            "source_metadata": doc.get("metadata", {}),
            **(extra_metadata or {})  # Optional additional info
        }
        documents.append(chunk)

    # Initialize or update FAISS vectorstore
    if vectorstore is None:
        logging.info("Initializing new FAISS vectorstore")
        vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=[doc.metadata for doc in documents])
    else:
        logging.info("Adding to existing FAISS vectorstore")
        vectorstore.add_texts(texts, metadatas=[doc.metadata for doc in documents])

# --- Load a saved FAISS index from disk ---
def load_index(path="faiss_index"):
    global vectorstore
    logging.info(f"Loading FAISS index from {path}")
    vectorstore = FAISS.load_local(path, embedding_model)

# --- Save current FAISS index to disk ---
def save_index(path="faiss_index"):
    logging.info(f"Saving FAISS index to {path}")
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

# --- Get a retriever interface for querying ---
def get_retriever():
    """
    Returns a retriever object for similarity-based search.
    Uses MMR (Maximal Marginal Relevance) to diversify results.
    """
    if vectorstore is None:
        logging.error("Attempted to retrieve from uninitialized vectorstore")
        raise ValueError("Vectorstore not initialized")
    logging.info("Returning vectorstore retriever")
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 25, 'lambda_mult': 0.4})
