import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

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

def main():
    # Load environment variables
    load_dotenv()

    # Initialize PDF loader
    loader = PyPDFLoader("docs/chicken.pdf")
    pages = loader.load_and_split()

    # Initialize ChatOpenAI model
    llm = ChatOpenAI(temperature=0)

    # Example questions
    question1 = "describe this document content ?"
    question2 = "how to make veg curry ?"
    question3 = "how to make airplane ?"

    # Execute chains for each question
    print(map_reduce_chain(llm, pages, question1))
    print(refine_chain(llm, pages, question2))
    print(stuff_chain(llm, pages, question3))

if __name__ == '__main__':
    main()
