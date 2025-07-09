import base64
import os
from mistralai import Mistral
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

def encode_file(file_path):
    """Encode the pdf or image to base64."""
    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def ocr_pdf(pdf_path: str, filename: str) -> list[Document]:
    """Extract pdf text via Mistral OCR"""

    # Getting the base64 string
    base64_pdf = encode_file(pdf_path)

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}" 
            },
            include_image_base64=True
        )

        # Converting ocr_reponse to markdown
        documents = []
        for i, page in enumerate(ocr_response.pages):
            documents.append(Document(
                page_content=str(page.markdown),
                metadata={
                    "source": filename,
                    "page": i+1,
                    # "element_type": element.category
                }
            ))
        return documents
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Create checking duplicate file uploads


def ocr_image(image_path: str):
    """Extract image text via Mistral OCR"""

    # Getting the base64 string
    base64_image = encode_file(image_path)

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}" 
            },
            include_image_base64=True
        )
        texts = []
        for page in ocr_response.pages:
            texts.append(page.markdown)
        extracted_text = " ".join(texts)
        return extracted_text

    except Exception as e:
        print(f"Image OCR processing failed: {e}")
        return None