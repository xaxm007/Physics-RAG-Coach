# Main Streamlit application
import os
import base64
from mistralai import Mistral
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import modules
from src.config import DB_FOLDER_PATH, PROMPT_TEMPLATE, CONTEXTUALIZE_PROMPT
from src.models import get_models
from src.document_loader import load_pdf, encode_image
from src.chunking import create_chunks
from src.database import create_db, load_db

# Initialize environment
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="Coach", page_icon="📚", layout="wide")
st.title("📚 The More You Know")
st.caption("Ask physics questions.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your Coach. Ask me anything about physics.")
    ]
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

# Initialize models based on user selection
llm_provider = st.sidebar.selectbox("LLM Provider", ["gemini", "mistral"], index=0)
try:
    embeddings, llm = get_models(llm_provider=llm_provider)
except Exception as e:
    st.error(f"Model initialization error: {str(e)}")
    st.stop()

with st.sidebar:

    st.divider()
    # Check if DB exists
    if os.path.exists(DB_FOLDER_PATH):
        st.success("Knowledge Base Ready")
        db = load_db(DB_FOLDER_PATH, embeddings)
        if db:
            try:
                count = db._collection.count()
                # st.info(f"Knowledge chunks: {count}")
            except Exception as e:
                st.info(f"Database info: {str(e)}")
    else:
        st.warning("Database not initialized")
    
    # PDF upload and processing
    if st.button("Initialize/Reinitialize Database"):

        uploaded_pdf_file = st.file_uploader("Upload new PDF to create knowledge base.", type="pdf", key="pdf_uploader")

        if uploaded_pdf_file is not None:

            os.makedirs('./pdf', exist_ok=True)
            temp_pdf_path = os.path.join('./pdf', uploaded_pdf_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf_file.getbuffer())
            st.success("PDF uploaded successfully!")
            pdf_path = temp_pdf_path

            if os.path.exists(pdf_path):
                with st.spinner("Processing PDF and creating database..."):
                    documents = load_pdf(
                        pdf_path=pdf_path,
                    )
                    
                    if documents:
                        chunks = create_chunks(documents)
                        db = create_db(DB_FOLDER_PATH, embeddings, chunks)
                        if db:
                            st.session_state.db_initialized = True
                            st.success("Database created successfully!")
        else:
            st.error("No PDF file found. Please upload one first.")

    st.divider()

    uploaded_image_file = st.file_uploader("Upload an Image for query", type=["png", "jpg", "jpeg"], key="image_uploader")

    if uploaded_image_file is not None:

        st.image(uploaded_image_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Extract Text from Image"):
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_api_key:
                st.error("MISTRAL_API_KEY not found in environment variables. Cannot perform OCR.")
            else:
                client = Mistral(api_key=mistral_api_key)
                base64_image = encode_image(uploaded_image_file)
                if base64_image:
                    with st.spinner("Extracting text from image ..."):
                        try:
                            ocr_response = client.ocr.process(
                                model="mistral-ocr-latest",
                                document={
                                    "type": "image_url",
                                    "image_url": f"data:{uploaded_image_file.type};base64,{base64_image}" 
                                },
                                include_image_base64=True
                            )
                            extracted_text = " ".join([page.markdown for page in ocr_response.pages])
                            st.session_state.extracted_image_text = extracted_text
                            st.success("Text extracted successfully!")
                        except Exception as e:
                            st.error(f"Image OCR processing failed: {str(e)}")

    st.divider()
    
    st.subheader("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = [
            AIMessage(content="Chat history cleared. Ask me a new question!")
        ]
        st.rerun()

# Main chat interface
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

            if hasattr(message, 'metadata') and 'sources' in message.metadata:
                with st.expander("Sources"):
                    for source in message.metadata['sources']:
                        source_label = os.path.basename(source.get("source", "Unknown"))
                        page = source.get("page", "N/A")
                        question_label = source.get("question_label", "")
                        references = source.get("references", "")
                        
                        source_text = f"📚 {source_label} (Page {page})"
                        if question_label:
                            source_text += f" | {question_label}"
                        if references:
                            source_text += f" | References: {references}"
                            
                        st.caption(source_text)

# Setup RAG system if DB exists
rag_chain = None
if os.path.exists(DB_FOLDER_PATH):
    db = load_db(DB_FOLDER_PATH, embeddings)
    if db:
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 3,
                'score_threshold': 0.55,
            },
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_TEMPLATE),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

# User input handling
user_query = st.chat_input("Ask a physics question...")

# Use extracted text as query if available
if user_query is None and "extracted_image_text" in st.session_state and st.session_state.extracted_image_text:
    user_query = st.session_state.extracted_image_text
    st.session_state.extracted_image_text = ""  # Clear after use

if user_query:
    
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Prepare chat history
    langchain_history = []
    for msg in st.session_state.chat_history[:-1]:
        langchain_history.append(msg)

    if rag_chain is not None:

        with st.spinner("Generating ..."):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                sources = []  # To collect source documents
                
                try:
                    inputs = {
                        "input": user_query,
                        "chat_history": langchain_history,
                        "context": []  # Will be filled by retriever
                    }
                    
                    # Stream the response
                    for chunk in rag_chain.stream(inputs):
                        if 'answer' in chunk:
                            full_response += chunk['answer']
                            message_placeholder.markdown(full_response + "▌")
                        # Collect source documents from context
                        if 'context' in chunk:
                            sources = [
                                {
                                    "source": doc.metadata.get("source", "Unknown"),
                                    "page": doc.metadata.get("page", "N/A"),
                                    "question_label": doc.metadata.get("question_label", ""),
                                    "references": doc.metadata.get("references", "")
                                }
                                for doc in chunk['context']
                            ]
                    
                    # Display final response
                    message_placeholder.markdown(full_response)
                    
                    # Create message with metadata for sources
                    ai_message = AIMessage(content=full_response)
                    ai_message.metadata = {"sources": sources}  # Attach sources as metadata
                    st.session_state.chat_history.append(ai_message)
                    
                    # Display sources immediately below the answer
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                source_label = os.path.basename(source.get("source", "Unknown"))
                                page = source.get("page", "N/A")
                                question_label = source.get("question_label", "")
                                references = source.get("references", "")
                                
                                source_text = f"📚 {source_label} (Page {page})"
                                if question_label:
                                    source_text += f" | {question_label}"
                                if references:
                                    source_text += f" | References: {references}"
                                    
                                st.caption(source_text)
                    else:
                        # with st.expander("Sources"):
                        st.caption("⚠️ No sources identified for this response")
                        st.caption("The answer was generated from general knowledge without specific document references")
                    
                except Exception as e:
                    error_msg = f"⚠️ Error processing your request: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))