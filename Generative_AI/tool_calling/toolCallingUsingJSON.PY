from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from dotenv import load_dotenv
load_dotenv()
query = "what is answer of multiply 12 with 18?"


llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5
)

parser=JsonOutputParser()
prompt = PromptTemplate(

        template=f"""
        You are a helpful assistant. 
     Given the user query : {query}, 
     Return your response as a JSON blob with 'name' and 'arguments' keys.""",
        input_variables=['query'],
        partial_variables={"format_instruction":parser.get_format_instructions()}

)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the product"""
    return a * b


chain=prompt|llm|parser|multiply
result=chain.invoke({"query":"what is output of 8 multiple by 3?"})
print(result)