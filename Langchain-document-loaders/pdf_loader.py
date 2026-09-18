from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# jab bhi documents loader karte hai to uska output list of documents me milta hai.
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following phrase - \n {phrase}',
    input_variables=['phrase']

)


loader = PyPDFLoader('D:\\GenAI_langchain\\Langchain-document-loaders\\Anveet-Pal-NR.pdf')

docs = loader.load()
print('--------------------------------')
print(len(docs))

print(docs[0].page_content)