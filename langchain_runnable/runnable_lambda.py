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
    template="Write a joke about {topic}",
    input_variables=['topic']
)




def word_count(text):
    return len(text.split())
parser=StrOutputParser()
joke=RunnableSequence(prompt1,llm1, parser)
parallel_chain = RunnableParallel(
    {
        "topic": RunnablePassthrough(), 
        "word_count": RunnableLambda(word_count)  ## this logic not work for joke count if we provide runnablsequence before thats why we use separate chain 
    }
)
chain=RunnableSequence(joke,parallel_chain)
result=chain.invoke({"topic":"AI"})
print(result)

 