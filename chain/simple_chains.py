from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  
)

chat_model = ChatHuggingFace(llm=llm)


prompt=PromptTemplate(
    template="generate five facts about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()
chain=prompt|chat_model|parser
result=chain.invoke({"topic":"AI"})
print(result)

print("CHAIN VISULIZATION")
print(chain.get_graph().print_ascii())