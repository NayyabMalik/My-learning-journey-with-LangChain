from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableSequence,RunnablePassthrough

load_dotenv()

llm1= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)




prompt1=PromptTemplate(
    template="Write a tweet about {topic}",
    input_variables=['topic']
)



parser=StrOutputParser()

parallel_chain=RunnableParallel(
{
    "topic":RunnablePassthrough(), # give us topc name as it is 
    "tweet": RunnableSequence(prompt1, llm1, parser),
}
)

result=parallel_chain.invoke({"topic":"AI"})
print(result)

 