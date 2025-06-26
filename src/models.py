# src/models.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from typing import Literal, Tuple
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

def get_models(llm_provider: Literal["gemini", "mistral"] = "gemini") -> Tuple[Embeddings, BaseChatModel]:
    """
    Initialize and return embeddings and LLM models based on selected provider
    
    Args:
        llm_provider: "gemini" for Google Gemini or "mistral" for Mistral AI
    
    Returns:
        Tuple of (embeddings, language_model)
    
    Raises:
        ValueError: If required API keys are missing
        RuntimeError: If model initialization fails
    """
    try:
        # Initialize embeddings (always using Google Gemini for consistency)
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize selected LLM
        if llm_provider == "gemini":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=None,
                timeout=120,
                max_retries=2,
                google_api_key=google_api_key
            )
            
        elif llm_provider == "mistral":
            mistral_api_key = os.getenv('MISTRAL_API_KEY')
            if not mistral_api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
                
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,
                max_retries=2,
                api_key=mistral_api_key
            )
            
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}. Choose 'gemini' or 'mistral'")
            
        return embeddings, llm
        
    except ValueError as ve:
        raise ve  # Re-raise validation errors
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {str(e)}")