from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

prompt=PromptTemplate(
    template="""Write a short, well-structured poem about the following text:

{topic}
At the end of the poem, clearly add:
"— Written by Nayyab Malik""",
    input_variables=['topic']
)


llm= ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)
parser=StrOutputParser()


text_loader = TextLoader(
    r"RAG\Document_Loaders\transcription.txt",
    encoding="latin-1" 
)
docs = text_loader.load()

chain=prompt|llm|parser
result=chain.invoke({"topic":docs[0].page_content})

print(result)