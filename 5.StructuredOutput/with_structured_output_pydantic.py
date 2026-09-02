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


class Review(BaseModel):
    key_themes: list[str] = Field(description="List of key themes mentioned in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review")
    pros: Optional[list[str]] = Field(description="List of pros mentioned in the review")
    cons: Optional[list[str]] = Field(description="List of cons mentioned in the review")
    name: Annotated[Optional[str], "Name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this. Review by John Doe
""")

print(result)
print(result.summary)
print(result.sentiment)
print(result.name)