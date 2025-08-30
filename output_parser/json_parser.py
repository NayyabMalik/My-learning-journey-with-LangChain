from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template1=PromptTemplate(
    template="give me a gender,city,review about character{name} in {format_instruction}",
    input_variables=['name'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)



chain=template1|chat_model|parser

result=chain.invoke({'name':'harry potter'})

print(result)