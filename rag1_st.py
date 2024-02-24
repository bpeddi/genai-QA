import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from io import BytesIO
from typing import Tuple, List
from pypdf import PdfReader
import re
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter

def refine_chain(llm, pages, question):
    """
    Refines the existing answer based on the provided context and question.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to refine the answer.

    Returns:
        The refined answer.
    """
    # Define the refine prompt template
    refine_prompt_template = """
    The original question is: \n {question} \n
    The provided answer is: \n {existing_answer}\n
    Refine the existing answer if needed with the following context: \n {context_str} \n
    Given the extracted content and the question, create a final answer.
    If the answer is not contained in the context, say "answer not available in context. \n\n
    """
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )

    # Define the initial question prompt template
    initial_question_prompt_template = """
    Answer the question as precise as possible using the provided context only. \n\n
    Context: \n {context_str} \n
    Question: \n {question} \n
    Answer:
    """
    initial_question_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=initial_question_prompt_template,
    )

    # Load the QA chain for refinement
    refine_chain = load_qa_chain(
        llm=llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=initial_question_prompt,
        refine_prompt=refine_prompt,
    )

    # Execute the refine chain
    refine_outputs = refine_chain({"input_documents": pages, "question": question})
    return refine_outputs['output_text']

def stuff_chain(llm, pages, question):
    """
    Retrieves an answer for the given question based on the provided context.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to answer.

    Returns:
        The answer to the question.
    """
    # Define the question prompt template
    question_prompt_template = """
    Use the following pieces of context to answer the question at the end. If you 
    don't know the answer, just say that you don't know, don't try to make up an 
    answer.
    Context: \n {context} \n
    Question: \n {query} \n
    Answer:
    """
    prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "query"]
    )

    # Load the QA chain for answering
    stuff_chain = load_qa_chain(
        llm=llm, chain_type="stuff", prompt=prompt, document_variable_name="context",
    )

    # Execute the QA chain for answering
    stuff_answer = stuff_chain(
        {"input_documents": pages, "query": question}, return_only_outputs=True
    )
    return stuff_answer["output_text"]

def map_reduce_chain(llm, pages, question):
    """
    Retrieves an answer for the given question based on the provided context using map-reduce approach.

    Args:
        llm: Language model instance.
        pages: List of pages from the document.
        question: Question to answer.

    Returns:
        The answer to the question.
    """
    # Define the question prompt template
    question_prompt_template = """
    Answer the question as precise as possible using the provided context. \n\n
    Context: \n {context} \n
    Question: \n {question} \n
    Answer:
    """
    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    # Define the combine prompt template
    combine_prompt_template = """Given the extracted content and the question, create a final answer.
    If the answer is not contained in the context, say "answer not available in context. \n\n
    Summaries: \n {summaries}?\n
    Question: \n {question} \n
    Answer:
    """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    # Load the QA chain for map-reduce approach
    map_reduce_chain = load_qa_chain(
        llm=llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        question_prompt=question_prompt,
        combine_prompt=combine_prompt,
    )

    # Execute the QA chain for map-reduce approach
    map_reduce_outputs = map_reduce_chain(
        {"input_documents": pages, "question": question}, return_only_outputs=True
    )
    return map_reduce_outputs["output_text"]

#Parse PDF file 
def parse_pdf(file: BytesIO) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output # type: ignore


def text_to_docs(text: List[str]) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=50,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for  chunk in chunks:
            doc = Document(
                page_content=chunk
            )
            doc_chunks.append(doc)
    return doc_chunks


# def get_documents_from_pdf(pdf_files):
#     text = []
#     for pdf_file in pdf_files:
#         text = parse_pdf(BytesIO(pdf_file.getvalue()))
#         # print(text)
#         text = text + (text_to_docs(text))
#     return text

def get_documents_from_pdf(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        text = parse_pdf(BytesIO(pdf_file.getvalue()))
        # if isinstance(text, str):
        #     text = [text]
        documents = documents + text_to_docs(text) # type: ignore
    return documents

def main():
#     st.set_page_config(
#     page_title='GenAI use cases',
#     layout="wide",
#     # initial_sidebar_state="expanded",
# )
    st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    # Load environment variables
    load_dotenv()
    st.title ( " Question Answering with Large Documents using LangChain ")
    st.subheader(" This notebook demonstrates how to build a question-answering (Q&A) system using LangChain (load_qa_chain) ")
    # Upload PDF files
    pdf_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    if pdf_files:
        documents = []
        documents = get_documents_from_pdf(pdf_files)
        # Print the documents (this will be displayed in the terminal if running locally)
        # print(documents[0].page_content)

        col1, col2 = st.columns(2)
        col1.subheader('Your Text ')
        # col1.text_area("",documents,height=400)
        col1.write(documents)
        col2.subheader('Your Chat')
        st.subheader("Ask your Question here ?")
        prompt = col2.text_input("how to make veg curry ?")
        
        if prompt:
            st.session_state["chat_history"] = prompt
            with col2.chat_message("user") : 
                st.write(prompt)
                llm = ChatOpenAI(temperature=0)
            with col2.chat_message("assistant"):
                answer=stuff_chain(llm, documents, prompt)
                st.write(answer)
                st.session_state["chat_history"] = answer
            chat = st.session_state.get("chat_history")
            print(type(chat))
        #     for i, message in enumerate(st.session_state.chat_history):
        #         print(i)
        #         print(message)
        # # Initialize PDF loader
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        #     chunk_overlap=5,
        # )

        # loader = PyPDFLoader("docs/chiken.pdf")
        # documents = loader.load_and_split()
        # print(type(documents))
        # print(documents)

        # chunks = text_splitter.split_text(documents)
        # print(type(chunks))
        # print(chunks)
        # # Initialize ChatOpenAI model
        

        # # Example questions
        # question1 = "how to make veg curry ?"
        # question2 = "how to make veg curry ?"
        # question3 = "how to make airplane ?"

        # # Execute chains for each question
        # print(map_reduce_chain(llm, documents, question1))
        
        # print(stuff_chain(llm, pages, question3))

if __name__ == '__main__':
    main()
