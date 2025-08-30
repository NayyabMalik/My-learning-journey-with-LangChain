from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
messages=[]

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the product"""
    return a * b

class MultiplyRequest(BaseModel):
    tool: str = Field(..., description="The tool to use, must be 'multiply'")
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

parser = PydanticOutputParser(pydantic_object=MultiplyRequest)

llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)




query = "what is answer of multiply 12 with 18?"
prompt = f"""
You are a helpful assistant. 
When asked a math question, output a JSON object that matches this schema:
{parser.get_format_instructions()}

Question: {query}
"""

messages.append(HumanMessage(content=query))

response = llm.invoke(prompt)
messages.append(response)
print("Raw response:", response.content)

parsed = parser.parse(response.content)
print("Parsed:", parsed)

if parsed.tool == "multiply":
    result = multiply.invoke({"a": parsed.a, "b": parsed.b})
    print("Tool result:", result)
    # instead of ToolMessage, just rephrase result back to LLM
    messages.append(HumanMessage(content=f"The tool '{parsed.tool}' returned: {result}. Please provide the final answer."))

llm_with_tool_response = llm.invoke(messages)
print("Final LLM response:", llm_with_tool_response.content)
