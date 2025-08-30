from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema,OutputFixingParser
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="fact 1",description="detail about fact 1"),
    ResponseSchema(name="fact 2",description="detail about fact 2"),
    ResponseSchema(name="fact 3",description="detail about fact 3")

]
parser=StructuredOutputParser.from_response_schemas(schema)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=chat_model)

template1=PromptTemplate(
    template="give me a 3 facts about {topic} in {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)




chain=template1|chat_model|fixing_parser

result=chain.invoke({'topic':'weak eye sight'})

print(result)