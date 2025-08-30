from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableSequence,RunnablePassthrough,RunnableBranch

load_dotenv()

llm= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)



prompt=PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=['topic']
)

prompt_sum=PromptTemplate(
    template="Write a summerize the {topic}",
    input_variables=['topic']
)


def word_count(text):
    return len(text.split())

parser=StrOutputParser()

chain1=prompt|llm|parser

joke=RunnableSequence(prompt,llm, parser)
parallel_chain = RunnableBranch(

        (RunnableLambda(lambda x:len(x.split())),prompt_sum|llm|parser),
        (RunnablePassthrough())
)
chain=chain1|parallel_chain
result=chain.invoke({"topic":"AI"})
print(result)

 