from docx import Document

def parse_docx(file_path):
    """
    Parses a .docx Word file and separates:
    - Text content (paragraphs)
    - Tables (flattened into readable sentences)
    
    Returns:
    - A string of paragraphs
    - A string of flattened table contents
    """
    doc = Document(file_path)

    text_blocks = []       # Stores plain paragraph text
    table_sentences = []   # Stores flattened table text

    # Iterate over all block-level elements: paragraphs and tables
    for block in iter_block_items(doc):
        if isinstance(block, str):  # Paragraphs are returned as strings
            clean = block.strip()
            if clean:
                text_blocks.append(clean)
        else:  # Tables are returned as actual table objects
            flat = flatten_docx_table(block)
            if flat:
                table_sentences.append(flat)

    # Return combined text and table strings
    return "\n\n".join(text_blocks), "\n\n".join(table_sentences)


def iter_block_items(doc):
    """
    Yields each block (paragraph or table) in the document in the order it appears.

    Paragraphs are returned as plain text.
    Tables are returned as docx.table.Table objects.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for element in doc.element.body:
        if element.tag.endswith("tbl"):  # Table block
            yield Table(element, doc)
        elif element.tag.endswith("p"):  # Paragraph block
            yield Paragraph(element, doc).text


def flatten_docx_table(table, section_title=None):
    """
    Converts a Word table to a flat, human-readable format using key-value pairs.

    Returns:
    - A string representation of the table
    - Optionally prepends a section title
    """
    rows = list(table.rows)

    # Return empty if there's no data or header
    if len(rows) < 2:
        return ""

    # Extract headers from the first row
    headers = [cell.text.strip() for cell in rows[0].cells]
    lines = [section_title] if section_title else []

    # Process remaining rows as data
    for row in rows[1:]:
        values = [cell.text.strip() for cell in row.cells]

        # Skip malformed rows
        if len(values) != len(headers):
            continue

        # Format each row as a string of "Header: Value" pairs
        entry = ", ".join(f"{headers[i]}: {values[i]}" for i in range(len(headers)))
        lines.append(entry)

    # Return joined lines
    return "\n".join(lines)
