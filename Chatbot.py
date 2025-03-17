import streamlit as st
import os
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

# Set page configuration
st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")

# Title
st.title("PDF Q&A Chatbot")

# Try to get API key from environment variable first
openai_api_key = os.getenv("OPENAI_API_KEY")

# If not found in environment, ask the user
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()
else:
    st.success("Using OpenAI API key from environment variables.")

# Initialize session state for chat history and processed status
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for model and retrieval settings
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select model:", ["gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    chunk_size = st.number_input("Chunk size (tokens):", min_value=100, max_value=1000, value=500)
    chunk_overlap = st.number_input("Chunk overlap (tokens):", min_value=0, max_value=200, value=50)
    k_value = st.number_input("Number of chunks to retrieve:", min_value=1, max_value=10, value=5)
    
    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize the OpenAI models with the provided API key
try:
    chat = ChatOpenAI(model_name=model_name,
                    temperature=temperature,
                    openai_api_key=openai_api_key)

    embedding = OpenAIEmbeddings(model='text-embedding-3-small',
                                openai_api_key=openai_api_key)
    
    # Define prompt templates
    PROMPT_RETRIEVING_S = '''You are a helpful assistant answering questions based on the provided document. 
    Answer the question using the provided context. If you cannot answer the question based on the context,
    say "I don't have enough information to answer this question based on the document." 
    
    Important: If the context contains relevant information but is incomplete, use what's available
    and indicate what additional information might be needed.
    
    Format your answer clearly and concisely, using markdown formatting where appropriate.'''

    PROMPT_TEMPLATE_RETRIEVING_H = '''This is the question:
    {question}

    This is the context from the document:
    {context}'''

    prompt_retrieving_s = SystemMessage(PROMPT_RETRIEVING_S)
    prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_RETRIEVING_H)

    chat_prompt_template_retrieving = ChatPromptTemplate([prompt_retrieving_s,
                                                        prompt_template_retrieving_h])

    # Initialize output parser
    str_output_parser = StrOutputParser()

    # Function to load and process PDF
    # Note the underscore prefix on _pdf_content to tell Streamlit not to hash this parameter
    @st.cache_resource
    def load_and_process_pdf(_pdf_content, _chunk_size, _chunk_overlap, _k_value):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(_pdf_content)
            temp_file_path = temp_file.name
        
        try:
            # Load PDF
            loader_pdf = PyPDFLoader(temp_file_path)
            docs_list = loader_pdf.load()
            
            # Better text splitter for PDF content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            docs_list_split = text_splitter.split_documents(docs_list)
            
            # Create vector store
            vectorstore = Chroma.from_documents(documents=docs_list_split,
                                            embedding=embedding,
                                            persist_directory="./pdf-storage")
            
            # Create retriever
            retriever = vectorstore.as_retriever(search_type="similarity",
                                                search_kwargs={'k': _k_value})
            
            return retriever
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    # Function to format retrieved documents 
    def format_docs(docs):
        return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

    # Chat interface
    def display_chat():
        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input for new question
        user_question = st.chat_input("Type your question about the document:")
        
        if user_question:
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Display user question
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Create chain
            chain = (
                {'context': st.session_state.retriever | format_docs,
                 'question': RunnablePassthrough()}
                | chat_prompt_template_retrieving
                | chat
                | str_output_parser
            )
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in chain.stream(user_question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF document to chat with", type="pdf")
    
    if uploaded_file is not None:
        # Process the PDF
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF... This may take a moment depending on the file size."):
                try:
                    # Save the retriever in session state so it persists
                    st.session_state.retriever = load_and_process_pdf(
                        uploaded_file.getbuffer(),
                        chunk_size,
                        chunk_overlap,
                        k_value
                    )
                    st.session_state.pdf_processed = True
                    st.success("PDF processed successfully! You can now ask questions about it.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        # If PDF has been processed, show chat interface
        if st.session_state.pdf_processed:
            display_chat()
    else:
        st.info("Please upload a PDF file to begin. Once processed, you can ask questions about its content.")

except Exception as e:
    st.error(f"Error initializing OpenAI models: {str(e)}")