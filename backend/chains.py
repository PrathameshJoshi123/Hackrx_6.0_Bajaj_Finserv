import os
import logging
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from vector_store import get_retriever  # Adjust import path if needed
from langchain_mistralai.chat_models import ChatMistralAI

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Initialize the Groq LLM model ---
try:
    logger.info("Initializing Groq LLM with model: llama3-8b-8192")
    llm = ChatMistralAI(
        model="mistral-small",             # Language model used for QA
        api_key=os.getenv("MISTRAL_API_KEY"),  # API key pulled from environment variable
        temperature=0.2                     # Controls creativity vs. factual response
    )
except Exception as e:
    logger.exception("Failed to initialize Groq LLM")  # Logs full traceback
    raise  # Propagate the exception

# --- Define the custom prompt template ---
CUSTOM_PROMPT_TEMPLATE = """
You are a highly accurate assistant designed to answer customer questions based **only** on the provided policy context.

YOUR TASK:
- Search the entire context carefully and repeatedly to find the exact, complete answer.
- Only return facts found **explicitly** in the context; do not guess, assume, or infer.
- Use clear, friendly, and simple language.
- Answer in **1 to 3 sentences maximum**.
- **Do not** include line breaks, bullet points, formatting symbols, or additional commentary.
- Include specific conditions or sections in the policy in detail for the answer.
- If the answer is not found in the context, say: "This information is not available in the provided policy context."
- Your goal is to extract accurate, complete answers in natural, helpful language.
- Support your ans with the facts and evidence mention it in ans

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
