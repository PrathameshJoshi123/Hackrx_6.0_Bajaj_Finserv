from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from schema.schema import AnswerResponse, QueryRequest
from chains import get_qa_chain
from services.document_parser import parse_document_from_url
from vector_store import add_to_index
from services.auth import verify_token

# Initialize FastAPI application
app = FastAPI()

# Configure CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"]   # Allow all headers
)

# Define POST endpoint at /hackrx/run that returns an AnswerResponse model
@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_query(
    request: QueryRequest,                 # Incoming request body with documents and questions
    token: str = Depends(verify_token)    # Dependency to verify authentication token
):
    try:
        start = time.time()  # Record start time for performance measurement

        # Step 1: Parse the document from the provided URL
        parsed_doc = parse_document_from_url(request.documents)

        # Step 2: Add the parsed document to the FAISS index
        # The document is indexed to enable similarity-based retrieval
        add_to_index(parsed_doc, doc_type=parsed_doc["type"])

        # Step 3: Load the question-answering chain
        qa_chain = get_qa_chain()

        # Step 4: Prepare list of question inputs and run them through the QA chain in batch
        questions_input = [{"input": q} for q in request.questions]
        results = qa_chain.batch(questions_input)

        # Extract and clean up answers from the QA chain's results
        answers = [res["answer"].strip() for res in results]

        # Print the total time taken to process the request
        print(time.time() - start)

        # Return the list of answers in the expected response model
        return AnswerResponse(answers=answers)

    except Exception as e:
        # Catch any exception, return HTTP 500 with error message
        raise HTTPException(status_code=500, detail=str(e))
