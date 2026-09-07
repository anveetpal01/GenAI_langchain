from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


# HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=2000,
)

model = ChatHuggingFace(llm=llm)


# Pydantic model
class Feedback(BaseModel):

    # sentiment ki value sirf in 3 mein se ek ho sakti hai
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="The sentiment of the text"
    )


# Pydantic output parser
parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)


# Prompt
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following Feedback:

{Feedback}

{format_instruction}
""",
    input_variables=['Feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)


# Chain
classfier_chain = prompt1 | model | parser2


# Invoke chain
result = classfier_chain.invoke({
    'Feedback': 'This is a worst smartphone'
})


# Access sentiment
print(result.sentiment)