# embedding models ka istemal ham text ko vector me convert karne ke liye use karte hai. 
# Taaki vector ke ander uss perticular text ka contextual understanding aajaye.

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of Frace"
]

result = embedding.embed_documents(documents)

print(str(result))

