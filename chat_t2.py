from langchain.chat_models import ChatOpenAI
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )
from langchain.memory import  ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain



load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# llm = ChatOpenAI(temperature=0)
model_id="google/flan-t5-large"
llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8, "max_new_tokens":200})
memory = ConversationBufferMemory()

conversation = ConversationChain ( llm = llm, memory=ConversationBufferMemory(memory_key='history', return_messages=True), verbose=False)


while True:
    user_input = input("> ")

    ai_response = conversation.predict(input=user_input)

    print("\nAssistant:\n", ai_response, "\n")
    print(conversation)
# messages = [
#     SystemMessage(content="You are a helpful assistant.")
# ]

# while True:
#     user_input = input("> ")
#     messages.append(HumanMessage(content=user_input))
#     assistant_response = chat(messages)
#     messages.append(AIMessage(content=assistant_response.content))
#     print("\nAssistant:\n", assistant_response.content, "\n")
#     print(messages)