from contextlib import asynccontextmanager
import json
import os
import logging
import tempfile
from typing import Annotated, List, Optional

from pydantic import BaseModel
from retrieval.rag_pipeline import rag_chain
from pinecone_db.pinecone_client import load_pinecone
from indexing.load_file import ocr_pdf, ocr_image
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # Logs to console
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    vector_store = load_pinecone()
    app.state.vector_store = vector_store
    print("Vector store loaded.")
    yield
    print("Shutting down...")
    if hasattr(vector_store, "close"):
        vector_store.close()
        print("Vector store closed.")

# MongoDB connection
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['Prompt_template']
collection = db['prompts']

app = FastAPI(
    title="Physics Coach Backend",
    description="An API for the Physics Coach Streamlit app.",
    lifespan=lifespan
)

origins = [
    'http://localhost:8501'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QAResponse(BaseModel):
    question: str
    answer: dict

@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "Physics Coach API is running"}


@app.post("/query", response_model=QAResponse)
async def query(
    request: Request,
    text_query: Annotated[str | None, Form(description="Enter your question")] = None,
    # history: Annotated[list[str], Form(description="Chat History")] = [],
    chat_history: Annotated[str, Form(description="Chat History")] = "[]",
    image_query: Annotated[UploadFile | bytes, File()] = None,
    prompt: Annotated[str, Form(description="Choose a prompt template")] = "general prompt"
):
    try:
        history = json.loads(chat_history)
    except json.JSONDecodeError:
        history = []
    try:
        if (text_query is None and image_query is None) or (text_query and image_query):
            raise HTTPException(
                status_code=400,
                detail="Provide either 'text_query' or 'image_query', not both"
            )
        
        if text_query:
            user_query = text_query
        else:
            # Process image with OCR
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                try:
                    content = await image_query.read()
                    temp_image.write(content)
                    temp_image_path = temp_image.name
                    user_query = ocr_image(temp_image_path)
                    logger.info(f"Processed image: {image_query.filename}")
                except Exception as e:
                    logger.error(f"Error processing image with OCR: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to process image: {str(e)}"
                    )
                finally:
                    os.unlink(temp_image_path)  # No need to check os.path.exists
        
        template = collection.find_one({"name": prompt})
        if template is None:
            raise HTTPException(status_code=404, detail=f"Prompt template '{prompt}' not found")
        formatted_template = template["template"]
        rephrase = collection.find_one({"name": "context prompt"})["template"]
        vector_store = request.app.state.vector_store
        response = rag_chain(user_query, history, vector_store, formatted_template, rephrase)
        return QAResponse(question=user_query, answer=response)
    
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

class PromptTemplate(BaseModel):
    name: str
    template: str

@app.get('/prompts', response_model=List[PromptTemplate])
async def get_prompts():
    prompts = list(collection.find({}, {"_id": False}))
    return prompts

@app.put('/prompts/{name}', response_model=PromptTemplate)
async def update_prompt(name: str, prompt: PromptTemplate):
    if prompt.name != name:
        raise HTTPException(status_code=400, detail="Prompt name in body must match URL parameter")
    result = collection.update_one(
        {"name": name},
        {"$set": {"template": prompt.template}},
        upsert=True
    )
    if result.matched_count == 0 and result.upserted_id is None:
        raise HTTPException(status_code=404, detail="Prompt not found and not created")
    return {"name": name, "template": prompt.template}




# @app.post("/query", response_model=QAResponse)
# async def query(
#     request:Request,
#     text_query: Optional[str] = Form(None),
#     history: List[str] = Form(...),
#     image_query: Optional[UploadFile] = File(None)
# ):
#     try:
#         if (text_query is None and image_query is None) or (text_query and image_query):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Provide either 'text_query' or 'image_query', not both"
#             )
        
#         if text_query:
#             user_query = text_query
#         else:
#             # Create temporary file
#             with tempfile.NamedTemporaryFile(delete=False) as temp_image:
#                 try:
#                     content = await image_query.read()
#                     temp_image.write(content)
#                     temp_image_path = temp_image.name
#                     user_query = ocr_image(temp_image_path)
#                     logger.info(f"Received file: {image_query.filename}")
#                     return user_query
#                 except Exception as e:
#                     logger.error(f"Error processing file: {str(e)}")
#                     raise HTTPException(
#                         status_code=500,
#                         detail=f"Failed to process image with OCR: {str(e)}"
#                     )
#                 finally:
#                     # Clean up temporary file
#                     if os.path.exists(temp_image_path):
#                         os.unlink(temp_image_path)

#             # contents = await image_query.read()
#             # user_query = ocr_image(contents)

#         vector_store = request.app.state.vector_store
#         response = rag_chain(user_query, history, vector_store)
#         # return {"question": user_query, "answer": response}
#         return QAResponse(question=user_query, answer=response)
    
#     except Exception as e:
#         print(f"Error processing query: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal server error processing your query"
#         )





# # Pydantic Model
# class QueryRequest(BaseModel):
#     text_query: str | None
#     history: list

# @app.post("/query")
# async def query(request: Request, body: QueryRequest):
#     try: 
#         if body.text_query is None and body.image_query is None:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Either text_query or image_query must be provided"
#             )
#         if body.text_query:
#             user_query = body.text_query

#         vector_store = request.app.state.vector_store
#         chat_history = body.history
#         response = rag_chain(user_query, chat_history, vector_store)
#         return {
#             "answer": response
#         }
#     except Exception as e:
#         print(f"Error processing query: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal server error processing your query"
#         )

# @app.post("/uploadpdf")
# async def upload_pdf(file: UploadFile | None = None):
#     """Upload PDF for Knowledge Base"""
#     if file.content_type != "application/pdf":
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF files are accepted"
#         )
    
#     # Create temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#         try:
#             content = await file.read()
#             temp_pdf.write(content)
#             temp_pdf_path = temp_pdf.name

#             documents = ocr_pdf(temp_pdf_path, file.filename)
#             logger.info(f"Received file: {file.filename}")

#             return {
#                  "filename": file.filename,
#                  "documents": documents
#             }
#         except Exception as e:
#             logger.error(f"Error processing file: {str(e)}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to process PDF with OCR: {str(e)}"
#             )

#         finally:
#             # Clean up temporary file
#             if os.path.exists(temp_pdf_path):
#                 os.unlink(temp_pdf_path)
