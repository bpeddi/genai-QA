from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

chat = ChatOpenAI(temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("> ")
    messages.append(HumanMessage(content=user_input))
    assistant_response = chat(messages)
    messages.append(AIMessage(content=assistant_response.content))
    print("\nAssistant:\n", assistant_response.content, "\n")
    print(messages)