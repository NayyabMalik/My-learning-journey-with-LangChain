from langchain_core.prompts import ChatPromptTemplate

# Define template with roles
chatTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are expert of {domain}."),
    ("human", "Explain in simple terms, what is {topic}?")
])

# Fill placeholders
prompt = chatTemplate.invoke({"domain": "ARTIFICIAL INTELLIGENCE", "topic": "Deep Learning"})

# Inspect the final prompt
print(prompt)
