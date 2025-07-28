import os
import requests 
from urllib.parse import urlparse
from pathlib import Path
from mimetypes import guess_extension, guess_type

# Import parsers for supported document types
from services.pdf_parser import parse_pdf
from services.docx_parser import parse_docx
from services.email_parser import parse_email

# Downloads a file from the provided URL and saves it to a temporary directory
def download_file(url, save_dir="/tmp/temp_downloads"):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Parse URL to extract filename or generate a fallback filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or f"downloaded_{str(hash(url))}"
    file_path = os.path.join(save_dir, filename)

    # Define headers to mimic a browser request (helps bypass some bot protections)
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Make the GET request to download the file
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        if not response.content:
            raise Exception("Download failed: File content is empty.")

        # Write the file in chunks to handle large files efficiently
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return file_path  # Return the local file path after download
    else:
        # Raise an exception if the download fails
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Parses a downloaded document based on its file type
def parse_document_from_url(url):
    # Step 1: Download the file and get its local path
    file_path = download_file(url)

    # Get the file extension (e.g., .pdf, .docx, .eml)
    ext = Path(file_path).suffix.lower()

    # Step 2: Parse based on file extension

    # PDF Handling
    if ext == ".pdf":
        text, tables = parse_pdf(file_path)  # Extract text and tables
        combined_text = text + "\n\n" + "\n\n".join(tables)  # Combine for consistency
        return {
            "type": "pdf",
            "text": combined_text,
            "metadata": {}  # No metadata for PDF
        }

    # DOCX Handling
    elif ext == ".docx":
        text, tables = parse_docx(file_path)
        combined_text = text + "\n\n" + "\n\n".join(tables)
        return {
            "type": "docx",
            "text": combined_text,
            "metadata": {}  # No metadata for DOCX
        }

    # EML (Email) Handling
    elif ext == ".eml":
        metadata, body_text, tables = parse_email(file_path)
        combined_text = body_text + "\n\n" + "\n\n".join(tables)
        return {
            "type": "email",
            "text": combined_text,
            "metadata": metadata  # Return email-specific metadata
        }

    else:
        # Unsupported file type
        raise ValueError(f"Unsupported file type: {ext}")
