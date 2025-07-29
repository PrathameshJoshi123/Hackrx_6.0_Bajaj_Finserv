import os
import logging
import re
from uuid import uuid4
from typing import List, Dict, Optional
from dotenv import load_dotenv

import numpy as np
from sklearn.preprocessing import normalize

import faiss
from huggingface_hub import InferenceClient
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.docstore import InMemoryDocstore

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()

# --- Custom HuggingFace Embeddings ---
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", hf_token: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=self.model_name, token=self.hf_token)
        logging.info(f"Initialized HuggingFaceAPIEmbeddings with model: {self.model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"Embedding {len(texts)} documents")
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        logging.info("Embedding single query")
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.client.feature_extraction(texts)
        except Exception as e:
            logging.error(f"HuggingFace API Error: {str(e)}")
            raise ValueError(f"HuggingFace API Error: {str(e)}")

embedding_model = HuggingFaceAPIEmbeddings()

# --- Text Splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
)

# --- PDF Text Cleaner ---
def clean_pdf_text(text: str) -> str:
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    
    def fix_spaced_chars(s: str) -> str:
        return re.sub(
            r'(?:\b(?:[A-Za-z]\s){2,}[A-Za-z]\b)',
            lambda m: m.group(0).replace(" ", ""),
            s
        )
    
    text = fix_spaced_chars(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- FAISS Vectorstore ---
vectorstore: Optional[FAISS] = None

# --- Add Document to Index ---
def add_to_index(doc: Dict, doc_type: str, extra_metadata: Optional[Dict] = None):
    global vectorstore
    logging.info(f"Adding document to index. Type: {doc_type}")
    
    cleaned_text = clean_pdf_text(doc["text"])
    chunks = splitter.create_documents([cleaned_text], metadatas=[doc.get("metadata", {})])

    texts = [chunk.page_content for chunk in chunks]
    metadatas_for_faiss = []

    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = str(uuid4())
        chunk.metadata["doc_type"] = doc_type
        if extra_metadata:
            chunk.metadata.update(extra_metadata)
        metadatas_for_faiss.append(chunk.metadata)

    logging.info(f"Split document into {len(texts)} chunks")

    # --- Embed and Normalize ---
    embeddings = embedding_model.embed_documents(texts)
    norm_embeddings = normalize(np.array(embeddings), axis=1).astype("float32")

    if vectorstore is None:
        logging.info("Initializing new FAISS vectorstore with IndexFlatIP")
        dim = norm_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(norm_embeddings)

        docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas_for_faiss)]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
        id_map = {i: str(i) for i in range(len(docs))}

        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=id_map
        )
    else:
        logging.info("Adding to existing FAISS vectorstore")
        start_idx = len(vectorstore.index_to_docstore_id)
        vectorstore.index.add(norm_embeddings)
        for i, (text, meta) in enumerate(zip(texts, metadatas_for_faiss), start=start_idx):
            doc_id = str(i)
            vectorstore.docstore._dict[doc_id] = Document(page_content=text, metadata=meta)
            vectorstore.index_to_docstore_id[i] = doc_id

# --- Save / Load Index ---
def save_index(path="/tmp/faiss_index"):
    logging.info(f"Saving FAISS index to {path}")
    os.makedirs(path, exist_ok=True)
    if vectorstore:
        vectorstore.save_local(path)
    else:
        logging.warning("No vectorstore to save.")

def load_index(path="/tmp/faiss_index"):
    global vectorstore
    logging.info(f"Loading FAISS index from {path}")
    vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

# --- Retriever ---
def get_retriever():
    if vectorstore is None:
        logging.error("Vectorstore not initialized")
        raise ValueError("Vectorstore not initialized")
    logging.info("Returning FAISS retriever with k=20")
    return vectorstore.as_retriever(search_kwargs={"k": 10})

