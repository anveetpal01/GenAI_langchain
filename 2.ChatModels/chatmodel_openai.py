from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
model = ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10) # different temperature gives different answer.
result = model.invoke("What is the capital of India")
print(result.content)

