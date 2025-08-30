
#Type Dict, LLM return JSON Format 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)
class Review(TypedDict):
    Summary:str
    Sentiment:Annotated[str,"Return class of sentiment as well"] ## if  model not give good response so used-- Sentiment:str
    pros:Annotated[Optional[list[str]],"write down all the pros inside the list"]
Struct_model=chat_model.with_structured_output(Review)
result=Struct_model.invoke("NO one taught this concept on youtube with this much clarity you are the best teacher i have got thank you is not enough but whatever knowledge i have about ai and ml its only because of you ! stay always blissful")
print(result)