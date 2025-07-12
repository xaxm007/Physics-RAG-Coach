import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory

r = redis.Redis(
    host='redis-17940.c277.us-east-1-3.ec2.redns.redis-cloud.com',
    port=17940,
    decode_responses=True,
    username="default",
    password="JBPG48ft4GFaR1CNm5XtOF34Sn2Z3GwO",
)

