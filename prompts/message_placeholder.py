from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
chat_template=ChatPromptTemplate(
    [
        ('system','you are a helpful customer support agent'),
        MessagesPlaceholder(variable_name="chat_History"),
        ('human','{query}'),
    ]
)
chat_history=[]
with open('prompts\chat_History.txt') as f:
   chat_history.append(f.readline())
prompt=chat_template.invoke({"chat_History":chat_history,"query":"where are you going last night?"})
print(prompt)