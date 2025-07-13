import re
from langchain_core.documents import Document

def get_reference_question(text: str):
    """Get reference to which points to other question for answer. (eg. Please refer to 2070 Supp (Set A) Q.No. 9 d)"""
    match = re.search(
        r"Please refer(?:red)? to (20\d{2}[\s\w\.\(\)]*?Q\.?[\s]?No\.?[\s]?\d+\s?[a-zA-Z]?)",
        text,
        re.IGNORECASE
    )
    return match.group(1).strip() if match else None

def get_chunk(text: str, page_num: int, filename: str) -> list[Document]:
    """Create chunks based on the question labels. (eg. 47. 2067 Supp Q.No. 9 d)"""
    pattern = re.compile(
        r"(\d+\.\s*20\d{2}[\s\w\.\(\)]*?Q\.?\s?No\.?\s?\d+\s?[a-zA-Z]?)",
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        if i+1 < len(matches):
            end = matches[i+1].start()
        else:
            end = len(text)

        chunk_text = text[start:end].strip()
        question_header = matches[i].group(0).strip()
        reference_question = get_reference_question(chunk_text)

        chunks.append(Document(
            page_content=chunk_text,
            metadata = {
                "question": question_header,
                "reference": reference_question or "", # Pinecone does not allow None or null metadata
                "page": page_num,
                "source": filename
            }
        ))
    return chunks

def create_chunks(documents: list[Document]):
    """Create chunks for each question and its reference question if available."""
    chunks = []
    for i, page in enumerate(documents["documents"]):
        text = page["page_content"]
        page_num = page["metadata"]["page"]
        source = page["metadata"]["source"]
        chunk = get_chunk(text, page_num, source)
        chunks.extend(chunk)
    return chunks
