
#TJSON SCHEMA IS N0T  supported by hugging face model so we used output parser rather than structure output , LLM return JSON Format 

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
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


Struct_model=chat_model.with_structured_output(json_schema)
result=Struct_model.invoke("NO one taught this concept on youtube with this much clarity you are the best teacher i have got thank you is not enough but whatever knowledge i have about ai and ml its only because of you ! stay always blissful")
print(result)