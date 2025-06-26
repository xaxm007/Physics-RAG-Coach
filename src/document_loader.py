import base64
from mistralai import Mistral
from unstructured.partition.md import partition_md
from langchain_core.documents import Document
from src.config import MISTRAL_API_KEY

def load_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        raise IOError(f"PDF loading failed: {str(e)}")

    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not set")
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
            include_image_base64=True
        )
        return process_ocr_response(ocr_response, pdf_path)
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {str(e)}")

def process_ocr_response(ocr_response, pdf_path):
    documents = []
    for i, page in enumerate(ocr_response.pages):
        elements = partition_md(text=page.markdown)
        for element in elements:
            documents.append(Document(
                page_content=str(element),
                metadata={"page": i + 1, "source": pdf_path, "element_type": element.category}
            ))
    return documents

def encode_image(uploaded_file):
    try:
        if uploaded_file.size > 5 * 1024 * 1024:
            raise ValueError("Image size exceeds 5MB limit")
        return base64.b64encode(uploaded_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Image encoding failed: {str(e)}")