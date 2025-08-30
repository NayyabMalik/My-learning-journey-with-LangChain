from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda
load_dotenv()

llm= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)
class feedback(BaseModel):
    sentiment:Literal["negative","positive"]=Field(description="give sentiment of text")
parser=PydanticOutputParser(pydantic_object=feedback)



prompt=PromptTemplate(
    template="classify the text {text} in the negative or positive class \n {format_instruction}",
    input_variables=['text'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

prompt1=PromptTemplate(
    template="write a response for positive feedback /n {feedback}",
    input_variables=['feedback']
)

prompt2=PromptTemplate(
    template="write a response for negative feedback /n {feedback}",
    input_variables=['feedback']
)


classifier_chain=prompt|llm|parser
parser_str=StrOutputParser()
chain_post=prompt1|llm|parser_str
chain_neg=prompt2|llm|parser_str

branch_Chain=RunnableBranch(
  (lambda x:x.sentiment=="positive" ,chain_post),  # if condition for positive 
  (lambda x:x.sentiment=="negative",chain_neg),    # if condition for negative 
  RunnableLambda(lambda x:"could not find sentiment")               #else condition

)

chain=classifier_chain|branch_Chain
result=chain.invoke({"text":"this video is too bad just waste of time"})
print(result)