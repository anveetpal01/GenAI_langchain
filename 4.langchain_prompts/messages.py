from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from chatbot import model

load_dotenv()

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Lanchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.text))

print(messages)