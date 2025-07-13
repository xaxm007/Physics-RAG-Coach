import json
import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage

                
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="Coach", page_icon="📚")
st.title("📚 The More You Know")
st.caption("Ask physics questions.")

@st.cache_data
def load_prompt_templates():
    try:
        response = requests.get(f"{FASTAPI_URL}/prompts")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading prompt templates: {e}")
        return []

templates = load_prompt_templates()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prevent multiple submissions or input
if "processing" not in st.session_state:
    st.session_state.processing = False


# --- Sidebar image uploader ---
with st.sidebar:
    image_question = None
    st.markdown("### 📤 Upload an image instead of typing:")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], disabled=st.session_state.processing)
    if uploaded_image:
        st.image(uploaded_image)
        image_question = st.button("Query", disabled=st.session_state.processing)

    if templates:
        names = [template["name"] for template in templates]
        selected_name = st.selectbox(
            "Select a Prompt Template",
            options=names,
            placeholder="Select a template...",
            index=None
        )

        if selected_name:
            selected_template = next((item['template'] for item in templates if item["name"] == selected_name), None)
            # template = st.code(selected_template, language="markdown", height="content", width="stretch")
            template = st.text_area(
                "Prompt Template",
                value=selected_template, 
                disabled=True, 
                height=250,
                width="stretch"
            )

# Display chat history/messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

            if "sources" in message.metadata:
                with st.expander("📖 Sources"):
                    for src in message.metadata["sources"]:
                        source = src.get("source", "Unknown")
                        page = src.get("page", "N/A")
                        question = src.get("question", "")
                        reference = src.get("references", "")
                        source_text = f"**🆀 {question}**"
                        # if reference:
                        #     source_text += f" | 🔗 Reference: `{reference}`"
                        source_text += f" | 📄 *Page {int(page)} — `{source}`*"
                        st.caption(source_text)

question = st.chat_input("Ask a question:", disabled=st.session_state.processing)
if question or image_question:

    

    if (question is None and image_question is None) or (question and image_question):
        st.error("Provide either 'Text Query' or 'Image Query', not both", icon="🚨")
        # st.stop()
        st.session_state.processing = False
    else:
        st.session_state.processing = True  # ⛔ Disable inputs
        
    # Add last two chat messages as history
    langchain_history = []
    for msg in st.session_state.chat_history[-2:]:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        langchain_history.append({"role": role, "content": msg.content})

    with st.spinner("Generating ..."):
        try:
            if question:
                data = {"text_query": question, "chat_history": json.dumps(langchain_history), "prompt": selected_name}
                response = requests.post(
                    f"{FASTAPI_URL}/query",
                    data=data,
                )
            elif image_question:
                files = {"image_query": uploaded_image}
                data = {"chat_history": json.dumps(langchain_history), "prompt": selected_name}
                response = requests.post(f"{FASTAPI_URL}/query", data=data, files=files)

            response.raise_for_status()
            result = response.json()

            with st.chat_message("user"):
                query = result["question"]
                st.markdown(query)

            with st.chat_message("assistant"):
                sources = []
                for doc in result["answer"]["context"]:
                    sources.append({
                        "source": doc["metadata"].get("source", "Unknown"),
                        "page": doc["metadata"].get("page", "N/A"),
                        "question": doc["metadata"].get("question", ""),
                        "references": doc["metadata"].get("references", "")
                    })
                full_response = result["answer"]["answer"]
                st.markdown(full_response)

                # Display sources for response
                with st.expander("📖 Sources"):
                    for src in sources:
                        source = src.get("source", "Unknown")
                        page = src.get("page", "N/A")
                        question = src.get("question", "")
                        reference = src.get("references", "")
                        source_text = f"**🆀 {question}**"
                        # if reference:
                        #     source_text += f" | 🔗 Reference: `{reference}`"
                        source_text += f" | 📄 *Page {int(page)} — `{source}`*"
                        st.caption(source_text)
                        # st.markdown(f"> {content}")
                        # st.markdown("---")
                        

                # Add QA to history
                st.session_state.chat_history.append(HumanMessage(content=query))
                ai_message = AIMessage(content=full_response)
                ai_message.metadata = {"sources": sources}
                st.session_state.chat_history.append(ai_message)

        except requests.RequestException as e:
            st.error(f"Error connecting to the API: {str(e)}")
            st.session_state.processing = False

        finally:
            st.session_state.processing = False


# import streamlit as st
# import requests
# from langchain_core.messages import AIMessage, HumanMessage

# st.set_page_config(page_title="Coach", page_icon="📚")
# st.title("📚 The More You Know")
# st.caption("Ask physics questions.")

# question = st.chat_input("Ask a question:")

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hello! I'm your Coach. Ask me anything about physics.")
#     ]

# # Display chat history/messages
# for message in st.session_state.chat_history:
#     if isinstance(message, HumanMessage):
#         with st.chat_message("user"):
#             st.markdown(message.content)
#     elif isinstance(message, AIMessage):
#         with st.chat_message("assistant"):
#             st.markdown(message.content)

# if question:
#     with st.chat_message("user"):
#         st.markdown(question)

#     # Add last two chat messages as history
#     langchain_history = []
#     for msg in st.session_state.chat_history[-2:]:
#         role = "human" if isinstance(msg, HumanMessage) else "ai"
#         langchain_history.append((role, msg.content))

#     try:
#         response = requests.post("http://localhost:8000/query", json={"question": question, "history": langchain_history})
#         response.raise_for_status()
#         result = response.json()

#         with st.spinner("Generating ..."):
#             with st.chat_message("assistant"):
#                 full_response = result["answer"]
#                 st.markdown(full_response)
        
#         # Add QA to history
#         st.session_state.chat_history.append(HumanMessage(content=question))
#         st.session_state.chat_history.append(AIMessage(content=full_response))

#     except requests.RequestException as e:
#         st.error(f"Error connecting to the API: {str(e)}")