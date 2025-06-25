import os
import re
import base64
import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from unstructured.partition.md import partition_md
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="Coach", page_icon="📚", layout="wide")
st.title("📚 The More You Know")
st.caption("Ask physics questions.")

# Function definitions
def get_models():
        
    google_api_key = os.getenv('GOOGLE_API_KEY')
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )
    
    # api_key = os.getenv('MISTRAL_API_KEY')
    # llm = ChatMistralAI(
    #     model="mistral-large-latest",
    #     temperature=0,
    #     max_retries=2,
    #     api_key=api_key
    # )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return embeddings, llm

def load_document(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("MISTRAL_API_KEY not found in environment variables")
        return None

    client = Mistral(api_key=api_key)

    with st.spinner("Processing PDF with Mistral OCR..."):
        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=True
            )
        except Exception as e:
            st.error(f"OCR processing failed: {str(e)}")
            return None

    documents = []

    for i, page in enumerate(ocr_response.pages):
        # Partition the markdown text directly
        elements = partition_md(text=page.markdown)
        
        for element in elements:
            documents.append(Document(
                page_content=str(element),  # element is an UnstructuredElement
                metadata={"page": i + 1, "source": pdf_path , "element_type": element.category}
            ))

    return documents

def extract_reference_question(text: str) -> str | None:
    match = re.search(
        r"Please refer(?:red)? to (20\d{2}[\s\w\.\(\)]*?Q\.?[\s]?No\.?[\s]?\d+\s?[a-zA-Z]?)",
        text,
        re.IGNORECASE
    )
    return match.group(1).strip() if match else None

def chunk_physics_questions(text: str, page_num: int) -> list[Document]:
    pattern = re.compile(
        r"(\d+\.\s*20\d{2}[\s\w\.\(\)]*?Q\.?\s?No\.?\s?\d+\s?[a-zA-Z]?)",
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    documents = []

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()
        question_header = matches[i].group(0).strip()
        referenced_question = extract_reference_question(chunk_text)

        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "source": pdf_path,
                "page": page_num,
                "question_label": question_header,
                "references": referenced_question
            }
        ))
    return documents

def create_chunk(documents):
    # headers_to_split_on = [
    #     ("#", "Header 1"),
    #     ("##", "Header 2"),
    #     ("###", "Header 3"),
    # ]

    # markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    # split_documents = []
    # for doc in markdown_documents:
    #     splits = markdown_splitter.split_text(doc.page_content)
    #     for split in splits:
    #         new_doc = split
    #         new_doc.metadata.update(doc.metadata)
    #         split_documents.append(new_doc)

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,  # Reduced overlap for better chunking
    #     length_function=len,
    #     add_start_index=True,
    # )

    # chunks = text_splitter.split_documents(split_documents)
    
    all_chunks = []

    for i, page in enumerate(ocr_response.pages):
        page_chunks = chunk_physics_questions(page.markdown, page_num=i + 1)
        all_chunks.extend(page_chunks)

    return chunks

def create_db(db_folder_path, embeddings, chunks):
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_folder_path,
        )
        return db
    except Exception as e:
        st.error(f"Error creating database: {str(e)}")
        return None

def load_db(db_folder_path, embeddings):
    try:
        db = Chroma(persist_directory=db_folder_path, embedding_function=embeddings)
        return db
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None

# New function to encode image to base64
def encode_image(uploaded_file):
    """Encode the uploaded image file to base64."""
    try:
        if uploaded_file is not None:
            return base64.b64encode(uploaded_file.read()).decode('utf-8')
        return None
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None


# Configuration paths
db_folder_path = 'chroma_db2'
# markdown_path = './data/output.md'
# markdown_folder = './data'
pdf_path = './pdf/physics.pdf'

# Initialize models
embeddings, llm = get_models()
if embeddings is None or llm is None:
    st.error("Failed to initialize models. Check your API keys.")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your Coach. Ask me anything about physics.")
    ]
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

with st.sidebar:

    # PDF upload and processing
    uploaded_pdf_file = st.file_uploader("Upload new PDF", type="pdf", key="pdf_uploader")
    
    if uploaded_pdf_file is not None:

        os.makedirs('./pdf', exist_ok=True)
        temp_pdf_path = os.path.join('./pdf', uploaded_pdf_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf_file.getbuffer())
        st.success("PDF uploaded successfully!")
        pdf_path = temp_pdf_path
    
    # if st.button("Initialize/Reinitialize Database"):
        if os.path.exists(pdf_path):
            with st.spinner("Processing PDF and creating database..."):
                documents = load_document(
                    pdf_path=pdf_path,
                )
                
                if documents:
                    chunks = create_chunk(documents)
                    db = create_db(db_folder_path, embeddings, chunks)
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
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                st.error("MISTRAL_API_KEY not found in environment variables. Cannot perform OCR.")
            else:
                client = Mistral(api_key=api_key)
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
    
    # Check if DB exists
    if os.path.exists(db_folder_path):
        st.success("Knowledge Base Ready")
        db = load_db(db_folder_path, embeddings)
        if db:
            try:
                count = db._collection.count()
                # st.info(f"Knowledge chunks: {count}")
            except Exception as e:
                st.info(f"Database info: {str(e)}")
    else:
        st.warning("Database not initialized")
    
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
if os.path.exists(db_folder_path):
    db = load_db(db_folder_path, embeddings)
    if db:
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 3,
                'score_threshold': 0.55,
            },
        )


        PROMPT_TEMPLATE = """
        ```
        You are a physics tutor assistant. When given a user question, you will be provided with a relevant  context to the user's question. Your job is to answer based on the provided context.

        Instruction:
        1. Use the question type to determine response style:
        - **Short / Factual**: If the question asks for a definition, property, or brief fact, answer in 1–2 sentences.
        - **Explanatory / Conceptual**: If the question asks for explanation, reasoning, or conceptual understanding, answer in 3–5 sentences with clear, coherent explanation.
        - **Derivation / Proof**: If the question asks for a derivation or proof, present step-by-step logical derivation using equations or logical steps, and conclude with the final result.
        - **Numerical / Calculation**: If the question asks for a numerical solution or problem-solving, show the step-by-step calculation, include formulae, substitute values, compute intermediate steps, and give the final numerical answer with units.
        2. Identify the question type automatically from the phrasing and keywords:
        - Look for words like "define", "what is", "state" → Short.
        - Look for words like "explain", "why", "how does" → Explanatory.
        - Look for words like "derive", "show that", "prove" → Derivation.
        - Look for numerical values, ask for "calculate", "compute", "find the value" → Numerical.
        3. Maintain a clear, educational tone and ensure each step is understandable.

        ---

        Context:
        ```

        {context}

        ```

        User Question:
        ```

        {input}

        ````

        Answer:"```

        ````

        """



        contextualize_q_system_prompt = """Given chat history and the latest question, \
        create a standalone question that includes necessary context. \
        Do NOT answer the question, just reformulate it if needed."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
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
                
            except Exception as e:
                error_msg = f"⚠️ Error processing your request: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))