from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="give me a detail report on {topic}",
    input_variables=['topic']
)
template2=PromptTemplate(
    template="give me a summary  on {text}",
    input_variables=['text']
)

parser=StrOutputParser()
chain=template1|chat_model|parser|template2|chat_model|parser

result=chain.invoke({'topic':'Artificial Intelligence'})

print(result)