import os
from pinecone import Pinecone
from dotenv import load_dotenv
from models.utils import embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

def upsert_documents(chunks: list[Document]):
    """Upsert the chunks to Pinecone"""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    embedding_fn = embeddings()
    vector_store = PineconeVectorStore(index=index, embedding=embedding_fn)
    vector_store.add_documents(chunks)

def load_pinecone():
    """Load Pinecone"""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    embedding_fn = embeddings()
    vector_store = PineconeVectorStore(index=index, embedding=embedding_fn, namespace='pdf/physics.pdf')
    return vector_store
