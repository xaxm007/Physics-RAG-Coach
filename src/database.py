from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

def create_db(db_folder_path: str, embeddings: Embeddings, chunks: list):
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_folder_path,
        )
    except Exception as e:
        raise RuntimeError(f"Database creation failed: {str(e)}")

def load_db(db_folder_path: str, embeddings: Embeddings):
    try:
        return Chroma(
            persist_directory=db_folder_path,
            embedding_function=embeddings
        )
    except Exception as e:
        raise RuntimeError(f"Database loading failed: {str(e)}")