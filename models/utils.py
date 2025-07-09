import os
from typing import Literal, Tuple
from langchain_core.embeddings import Embeddings
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def embeddings(embed_provider: str = "mpnet"):
    """Initialize embedding model"""
    try:
        if embed_provider == "gemini":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            embedding_fn = GoogleGenerativeAIEmbeddings(
                model='models/embedding-001',
                google_api_key=google_api_key
            )

        elif embed_provider == "mpnet":
            embedding_fn = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-mpnet-base-v2",
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )
        return embedding_fn
    except Exception as e:
        raise f"Error: {str(e)}"
    

def chat_llm(llm_provider: str = "gemini"):
    """Initialize chat model"""
    try:
        if llm_provider == "gemini":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
                google_api_key=google_api_key,
            )
        elif llm_provider == "mistral":
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,
                max_retries=2,
                api_key=mistral_api_key
            )
        return llm
    except Exception as e:
        raise f"Error: {str(e)}"

# def get_models(embed_provider: Literal["gemini", "mpnet"] = "mpnet", llm_provider: Literal["gemini", "mistral"] = "gemini") -> Tuple[Embeddings, BaseChatModel]:
#     """Initialize and return embeddings and LLM models based on selected provider
#     Args:
#         llm_provider: "gemini" for Google Gemini or "mistral" for Mistral AI
#         embed_provider: "gemini" for Google Gemini or "mpnet" for Sentence Transformer from HuggingFace"""
#     embedding_fn = embeddings(embed_provider)
#     llm = chat_llm(llm_provider)

#     return embedding_fn, llm