from langchain import hub
from operator import itemgetter
from models.utils import chat_llm
from langchain.chains import LLMChain
from pinecone_db.pinecone_client import load_pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain

def rag_chain(user_query: str, chat_history: list, vector_store, template: str, rephrase: str):
    """RAG chain"""

    # # Initialize Pinecone database
    # vector_store = load_pinecone()

    # # Prompt
    # template = """Answer the question based only on the following context"""
    # prompt = ChatPromptTemplate.from_template(template)

    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", """Given a chat history and the latest user question which might reference context in the chat history, 
            # formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
            # just reformulate it if needed and otherwise return it as is."""),
            ("system", rephrase),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", "Answer the question based only on the following context and chat history"),
            # ("system", "Context: {context}"),
            ("system", template),
            # ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ]
    )

    # Initialize ChatModel
    llm = chat_llm()
    retriever = vector_store.as_retriever()
    
    chat_retriever_chain = create_history_aware_retriever(llm, retriever, rephrase_prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(chat_retriever_chain, doc_chain)

    # chain = (
    #     {
    #         "context": retriever,
    #         "question": RunnablePassthrough()
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    answer = chain.invoke({
        "input": user_query, 
        "history": chat_history
    })

    return answer

# def rag(user_query: str):
#     """Initialize Retrival Pipeline"""
#     vector_store = load_pinecone()
#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={'k': 6, 'lambda_mult': 0.25}
#     )
#     result = rag_chain(retriever, user_query)
#     return result

    # Multiple Query Translation using llm
    # retriever_chain = MultiQueryRetriever.from_llm(
    #     retriever=retriever,
    #     llm=llm
    # )