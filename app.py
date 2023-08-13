import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from htmlTemplates import css,bot_template,user_template


def get_text_from_pdf(pdf_files):
    text= ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_chunk = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_chunk.split_text(text)
    return chunks

def get_vector_embeddings(text_chunk):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunk, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm= ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse the chat history list to display the latest question first
    reversed_chat_history = reversed(st.session_state.chat_history)
    
    latest_messages = list(reversed_chat_history)[:2]  # Get the last two messages   
    # Display the latest question and answer
    if len(latest_messages) >= 2:
        bot_message = latest_messages[0]
        user_message = latest_messages[1]

        st.write(user_template.replace("{{MSG}}", user_message.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", bot_message.content), unsafe_allow_html=True)
               

def main():
    # To load .env settings
    load_dotenv()
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None    
    
    #Setting the page Configuration
    st.set_page_config(page_title="Query Your PDF",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)

    st.header("Query Your PDFs :books:")

    
    user_question = st.text_input("Enter your Question here for the document:")
    
    # Initialize chat history and new conversation flag in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = get_conversation_chain(vector_store)
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_files= st.file_uploader("Upload PDF and click on Process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Loading..."):
                    # 1. Get the PDF text
                    raw_text = get_text_from_pdf(pdf_files)
                    # 2. Get the Text Chunks from PDFs
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Create Vector Store for these embeddings
                    vector_store = get_vector_embeddings(text_chunks)
                    # 4. Creating a conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
        
    if user_question:
            handle_userInput(user_question)
        
if __name__ == '__main__':
    main()