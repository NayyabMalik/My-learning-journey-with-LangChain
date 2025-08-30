import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Now they are available via os.environ
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")



# Choose any available model on OpenRouter
# e.g. "mistralai/mistral-7b-instruct" or "openchat/openchat-7b"
chat_model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",  
    max_tokens=512,
    temperature=0.2
)

# Define prompts
prompt1 = PromptTemplate(
    template="generate detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="generate five points summary for the following text: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# Chain definition
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

# Run the chain
result = chain.invoke({"topic": "AI"})
print(result)

print("CHAIN VISUALIZATION")
print(chain.get_graph().print_ascii())
