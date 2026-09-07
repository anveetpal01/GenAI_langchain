from langchain_openai import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, max_completion_tokens=10) # different temperature gives different answer.

result = model.invoke("What is the capital of India")

print(result.content)