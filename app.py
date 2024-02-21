import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts import  PromptTemplate

import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks,api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore,api_key):
    # qa_system_template = """Answer the question using the following contexts.
    # ----------------
    # {context}"""
    # messages = [
    #     SystemMessagePromptTemplate.from_template(qa_system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    # qa_system_prompt = ChatPromptTemplate.from_messages(messages)
    # qa = ConversationalRetrievalChain.from_llm(chat_model, vector_store.as_retriever(search_kwargs={
    #                                            "k": 8}), memory=memory, condense_question_prompt=condense_question_prompt,
    #                                             verbose=True, combine_docs_chain_kwargs={"prompt": qa_system_prompt})
    llm = ChatOpenAI(api_key=api_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True,
        
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response )
    st.session_state.chat_history = response['chat_history']
    # st.write(st.session_state.chat_history )

    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            # st.write(user_template.replace(
            #     "{{MSG}}", message.content), unsafe_allow_html=True)
            with st.chat_message("user"):
                st.write(message.content)
        else:
            # st.write(bot_template.replace(
            #     "{{MSG}}", message.content), unsafe_allow_html=True)
            with st.chat_message("assistant"):
                st.write(message.content)


def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    print(api_key)
    os.environ['OPENAI_API_KEY'] = api_key
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # template = """You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    #     Keep your answer short and to the point.
        
    #     The evidence are the context of the pdf extract with metadata. 
        
    #     Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
        
    #     Make sure to add filename and page number at the end of sentence you are citing to.
            
    #     Reply "Not applicable" if text is irrelevant.
    #     {chat_history}
    #     Human: {input}
    #     Chatbot:"""

    # template = """
    #         You are a helpful Assistant who answers to users questions based on multiple contexts given to you.
            
    #         CONTEXT:
    #          Keep your answer short and to the point.
        
    #         The evidence are the context of the pdf extract with metadata. 
        
    #         Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
        
    #         Make sure to add filename and page number at the end of sentence you are citing to.
            
    #         Reply "Not applicable" if text is irrelevant.
            
    #         QUESTION: 
    #         {query}

    #         CHAT HISTORY: 
    #         {chat_history}
            
    #         ANSWER:
    #         """

    # prompt = PromptTemplate(
    #     input_variables=["chat_history", "query"], template=template
    # )
    
    # memory = ConversationBufferMemory(memory_key="chat_history")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks,api_key)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,api_key)


if __name__ == '__main__':
    main()
