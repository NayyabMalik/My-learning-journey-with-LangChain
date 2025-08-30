from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnableSequence

load_dotenv()

llm= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)


prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['tppic']
)
parser=StrOutputParser()

chain=RunnableSequence(prompt,llm,parser)

result=chain.invoke({"topic":"AI"})

print(result)