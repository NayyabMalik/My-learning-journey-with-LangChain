from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import requests
from operator import itemgetter
from typing import Annotated
from langchain_core.tools import InjectedToolArg
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import os

load_dotenv()

# -------------------- TOOLS --------------------
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch the currency conversion factor between base currency and target currency"""
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")  # Replace with valid key
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        conversion_rate = data.get("conversion_rate")
        if conversion_rate is None:
            raise ValueError(f"Conversion rate not found in API response: {data}")
        return conversion_rate
    except (requests.RequestException, ValueError) as e:
        raise Exception(f"Failed to fetch conversion rate: {str(e)}")

@tool
def convert(base_currency_value: float, converting_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert base currency value into target value"""
    return base_currency_value * converting_rate

tools = [get_conversion_factor, convert]

# -------------------- TOOL CHAIN --------------------
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return chosen_tool.invoke(model_output["arguments"])  # Directly invoke the tool with arguments

# Wrap tool_chain in RunnableLambda to make it compatible with RunnableSequence
tool_chain_runnable = RunnableLambda(tool_chain)

# -------------------- MODEL --------------------
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_completion_tokens=512
)

# Define schema for tool call
class ToolCall(BaseModel):
    name: str = Field(description="The tool name to call")
    arguments: dict = Field(description="Arguments to pass to the tool")

parser = JsonOutputParser(pydantic_object=ToolCall)

prompt = PromptTemplate(
    template=(
        "You are a strict assistant. Based on the query, output a JSON object with the following schema:\n"
        "{format_instructions}\n\n"
        "Available tools:\n"
        "- get_conversion_factor(base_currency: str, target_currency: str) -> float: Fetches the conversion rate.\n"
        "- convert(base_currency_value: float, converting_rate: float) -> float: Converts currency using the rate.\n\n"
        "Examples:\n"
        "Query: 'What is the currency conversion factor between PKR and USD?'\n"
        "Output: {{'name': 'get_conversion_factor', 'arguments': {{'base_currency': 'PKR', 'target_currency': 'USD'}}}}\n"
        "Query: 'Convert 10 USD to PKR using the conversion rate 278.5'\n"
        "Output: {{'name': 'convert', 'arguments': {{'base_currency_value': 10, 'converting_rate': 278.5}}}}\n\n"
        "Query: {query}\n\n"
        "If the query involves multiple steps (e.g., fetching a rate and converting), output the tool call for the first step (get_conversion_factor)."
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain for invoking the model and parsing output
chain = prompt | llm | parser | tool_chain_runnable

# -------------------- TEST --------------------
messages = []
try:
    # Step 1: Get conversion factor
    question = HumanMessage("What is the currency conversion factor between PKR and USD. How much 10 USD equal to PKR?")
    messages.append(question)
    step1 = chain.invoke({"query": question.content})
    print("Step 1 model output:", step1)

    # Since the query involves two steps, assume step1 is for get_conversion_factor
    if isinstance(step1, dict) and step1.get("name") == "get_conversion_factor":
        conversion_rate = step1  # Already invoked by tool_chain_runnable
        print("Step 1 result (conversion factor):", conversion_rate)
    else:
        conversion_rate = step1  # If tool_chain_runnable returned the rate directly

    # Step 2: Perform conversion
    question2 = HumanMessage(f"Convert 10 USD to PKR using the conversion rate {conversion_rate}")
    messages.append(question2)
    step2 = chain.invoke({"query": question2.content})
    print("Step 2 model output:", step2)

    # Result is already the converted value
    print("Step 2 result (final conversion):", step2)

except Exception as e:
    print("Error:", str(e))