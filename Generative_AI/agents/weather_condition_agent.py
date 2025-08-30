from langchain_community.tools import DuckDuckGoSearchRun,tool
import requests
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os
load_dotenv()
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

search_tool=DuckDuckGoSearchRun()

# put your WeatherAPI key here
weather_api = os.getenv("WEATHER_API_KEY")
@tool
def get_weather_data(city: str) -> dict:
    """This function gets weather data for a given city"""
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={city}"
    response = requests.get(url)
    return response.json()

# response = get_weather_data("Islamabad")
# print(response)


## prompt 
prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose=True

)

response=agent_executor.invoke({"input":"find the weather of islamabad"})
print(response)