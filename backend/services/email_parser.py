import os 
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

# Function to parse an email (.eml) file and extract metadata, text content, and tables
def parse_email(file_path):
    # Open and parse the email file with default email policy
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Extract key metadata fields from the email
    metadata = {
        "From": msg["From"],
        "To": msg["To"],
        "Subject": msg["Subject"],
        "Date": msg["Date"]
    }

    html = ""   # To store HTML content if available
    plain = ""  # To store plain text content if available

    # Check if the email is multipart (contains both plain and HTML parts, attachments, etc.)
    if msg.is_multipart():
        for part in msg.walk():  # Iterate through each part of the email
            content_type = part.get_content_type()
            if content_type == "text/html" and not html:
                html = part.get_content()
            elif content_type == "text/plain" and not plain:
                plain = part.get_content()
    else:
        # If email is not multipart, check content type directly
        content_type = msg.get_content_type()
        if content_type == "text/html":
            html = msg.get_content()
        elif content_type == "text/plain":
            plain = msg.get_content()

    body_text = ""        # To hold cleaned text from HTML or plain text
    flattened_tables = [] # To store extracted table content

    # Prefer HTML content if available for richer structure
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Extract visible text from HTML, separated by newlines
        body_text = soup.get_text(separator="\n", strip=True)

        # Look for <table> elements in the HTML and process them
        for i, table in enumerate(soup.find_all("table")):
            flat_table = flatten_html_table(table, f"Table {i + 1}")
            if flat_table:
                flattened_tables.append(flat_table)

    # Fallback to plain text if HTML is not present
    elif plain:
        body_text = plain.strip()

    # Return extracted metadata, email body text, and any flattened tables
    return metadata, body_text, flattened_tables


# Helper function to convert an HTML table to a flat string format
def flatten_html_table(table, section_title=None):
    rows = table.find_all("tr")  # Get all rows of the table
    if not rows or len(rows) < 2:
        return None  # Skip tables without at least a header and one row

    # Extract headers from the first row
    headers = [cell.get_text(strip=True) for cell in rows[0].find_all(["th", "td"])]

    # Optionally include a section title at the start
    lines = [section_title] if section_title else []

    # Process each subsequent row to extract cell data
    for row in rows[1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]

        # Skip row if column count doesn't match headers (inconsistent table)
        if len(cells) != len(headers):
            continue

        # Combine header and cell data into a single readable line
        line = ", ".join(f"{headers[i]}: {cells[i]}" for i in range(len(headers)))
        lines.append(line)

    # Return the entire table as a flattened string
    return "\n".join(lines)
