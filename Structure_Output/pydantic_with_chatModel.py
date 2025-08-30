from langchain_openai import ChatOpenAI
from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini through OpenRouter (not Google Cloud)
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",  # available via OpenRouter
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Define schema
class Review(BaseModel):
    Summary: str = Field(description="Brief summary of the review")
    Sentiment: Literal['neg', 'pos'] = Field(description="Sentiment: 'neg' or 'pos'")
    pros: Optional[list[str]] = Field(description="Pros described in the review")
    cons: Optional[list[str]] = Field(description="Cons described in the review")

# Use structured output
Struct_model = llm.with_structured_output(Review)

text = "NO one taught this concept on youtube with this much clarity you are the best teacher i have got thank you is not enough but whatever knowledge i have about ai and ml its only because of you ! stay always blissful"

result = Struct_model.invoke(text)

print("Structured result:\n", result)
print("\nAs dict:\n", result.dict())
