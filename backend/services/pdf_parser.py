import pdfplumber

def parse_pdf(file_path):
    """
    Extracts non-table text and table content separately from a native PDF.

    Returns:
    - full_text: cleaned body text (excluding table content)
    - flattened_table_text: tables converted into readable key-value sentences
    """
    non_table_text_blocks = []       # Holds body text outside tables
    flattened_table_blocks = []      # Holds flattened tables

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. Extract all words from the page
            words = page.extract_words()

            # Identify table bounding boxes to separate out table content
            table_areas = [table.bbox for table in page.find_tables()]
            
            # Filter out words that lie within any table's bounding box
            non_table_words = [
                w for w in words if not is_in_any_bbox(w, table_areas)
            ]

            # Convert the remaining words into structured text
            page_text = words_to_text(non_table_words)
            if page_text.strip():
                non_table_text_blocks.append(f"[Page {i + 1}]\n{page_text.strip()}")

            # 2. Extract tables from the page and flatten them
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:  # Ensure table has headers and at least one row
                    flat = flatten_table(table, section_title=f"Table from Page {i + 1}")
                    flattened_table_blocks.append(flat)

    # Return concatenated body text and flattened tables
    return (
        "\n\n".join(non_table_text_blocks).strip(),
        "\n\n".join(flattened_table_blocks).strip()
    )

def is_in_any_bbox(word, bboxes):
    """
    Check if a word's bounding box lies within any of the provided table bounding boxes.
    Used to separate body text from tables.
    """
    x0, top, x1, bottom = float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom'])
    for bbox in bboxes:
        left, t, right, b = bbox
        if (left <= x0 <= right or left <= x1 <= right) and (t <= top <= b or t <= bottom <= b):
            return True
    return False

def words_to_text(words):
    """
    Convert a list of word dictionaries into text grouped by line based on vertical position.
    """
    from itertools import groupby
    lines = []
    # Group words with similar 'top' position (i.e., same line)
    for _, line_words in groupby(words, key=lambda w: round(w['top'], 1)):
        line = " ".join(w['text'] for w in line_words)
        lines.append(line)
    return "\n".join(lines)

def flatten_table(table, section_title=None):
    """
    Converts a 2D list (PDF table) into readable key-value format text.
    Handles missing values and skips malformed rows gracefully.
    """
    if not table or len(table) < 2:
        return ""

    headers = table[0]  # First row as header
    rows = table[1:]    # Remaining rows are data
    lines = [section_title] if section_title else []

    for row in rows:
        if not row or len(row) != len(headers):
            continue  # Skip rows that don’t match the number of headers
        try:
            # Create a readable line from header-value pairs
            line = ", ".join(
                f"{(headers[i] or '').strip()}: {(row[i] or '').strip()}" for i in range(len(headers))
            )
            lines.append(line)
        except Exception as e:
            print(f"⚠️ Skipped row due to error: {e}")
            continue

    return "\n".join(lines)
