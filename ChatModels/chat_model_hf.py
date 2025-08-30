from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
## getting this model using API  
## to used locally on machine ,use pipelines
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  
)

chat_model = ChatHuggingFace(llm=llm)
result=chat_model.invoke("hello")
print(result.content)