import os
import logging
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from vector_store import get_retriever  # Adjust import path if needed

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Initialize the Groq LLM model ---
try:
    logger.info("Initializing Groq LLM with model: llama3-8b-8192")
    llm = ChatGroq(
        model="llama3-8b-8192",             # Language model used for QA
        api_key=os.getenv("GROQ_API_KEY"),  # API key pulled from environment variable
        temperature=0.2                     # Controls creativity vs. factual response
    )
except Exception as e:
    logger.exception("Failed to initialize Groq LLM")  # Logs full traceback
    raise  # Propagate the exception

# --- Define the custom prompt template ---
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful assistant trained to answer customer questions based strictly on the policy context provided below. 

INSTRUCTIONS:
- Search exhaustivly in the given context for the question.
- Use simple, user-friendly language
- Strictly Answer in 1-3 sentences maximum
- Do not guess or add information not in the context
- Strictly search in the context many times.
- And Specify each detail about that question in answer
- Strictly not use new line characters or formatting symbols in answers
- Provide direct, helpful answers based solely on the provided context

Context:
---------
{context}

Q: {input}
A:
"""

# --- Create the prompt template object ---
try:
    logger.info("Creating prompt template")
    prompt = PromptTemplate(
        input_variables=["context", "input"],  # Must match placeholders in the template
        template=CUSTOM_PROMPT_TEMPLATE
    )
except Exception as e:
    logger.exception("Failed to create prompt template")
    raise

# --- Construct the full RetrievalQA chain ---
def get_qa_chain():
    """
    Builds a RetrievalQA chain that:
    - Retrieves relevant documents using a vector retriever
    - Feeds the documents and question into a prompt
    - Uses the Groq LLM to generate an answer
    """
    try:
        logger.info("Fetching retriever from vectorstore")
        retriever = get_retriever()  # Retrieves the FAISS retriever or equivalent

        logger.info("Creating combine_docs_chain with Groq LLM")
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,       # LLM to use for answering
            prompt=prompt  # Custom prompt with QA instructions
        )

        logger.info("Creating retrieval QA chain")
        return create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )
    except Exception as e:
        logger.exception("Failed to create QA chain")
        raise
