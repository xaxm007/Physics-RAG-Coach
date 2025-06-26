import re
from langchain_core.documents import Document

def extract_reference_question(text: str) -> str | None:
    match = re.search(
        r"Please refer(?:red)? to (20\d{2}[\s\w\.\(\)]*?Q\.?[\s]?No\.?[\s]?\d+\s?[a-zA-Z]?)",
        text,
        re.IGNORECASE
    )
    return match.group(1).strip() if match else None

def chunk_physics_questions(text: str, page_num: int, pdf_path: str) -> list[Document]:
    pattern = re.compile(
        r"(\d+\.\s*20\d{2}[\s\w\.\(\)]*?Q\.?\s?No\.?\s?\d+\s?[a-zA-Z]?)",
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    documents = []

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()
        question_header = matches[i].group(0).strip()
        referenced_question = extract_reference_question(chunk_text)

        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "source": pdf_path,
                "page": page_num,
                "question_label": question_header,
                "references": referenced_question
            }
        ))
    return documents

def create_chunks(ocr_response, pdf_path):
    all_chunks = []
    for i, page in enumerate(ocr_response.pages):
        page_chunks = chunk_physics_questions(page.markdown, i + 1, pdf_path)
        all_chunks.extend(page_chunks)
    return all_chunks