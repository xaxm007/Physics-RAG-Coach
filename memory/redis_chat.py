import redis
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    redis_url = os.getenv("REDIS_URL")
    return RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

def format_output(result: dict) -> dict:
    return {
        "output": result["answer"],
        "response": result
    }