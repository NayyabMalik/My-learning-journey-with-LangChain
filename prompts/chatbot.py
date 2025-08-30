from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)
chat_history=[
    SystemMessage("you are a helpful assistant")
]
while True:
    user_input=input("you:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    else:
        result=chat_model.invoke(chat_history)
        print("AI:",result.content)
        chat_history.append(AIMessage(content=result.content))
print(chat_history)
