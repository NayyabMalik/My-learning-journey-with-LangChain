from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load webpage
webbased_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Black_hole")
docs = webbased_loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Define LLM + prompt
prompt = PromptTemplate(
    template="Extract the most important information from this text:\n\n{text}",
    input_variables=["text"]
)

llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=512,
)
parser = StrOutputParser()

# Process each chunk
chain = prompt | llm | parser
summaries = [chain.invoke({"text": chunk.page_content}) for chunk in chunks]

# Merge summaries into one
final_summary = "\n".join(summaries)
print(final_summary)
