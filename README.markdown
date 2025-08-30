# LangChain Learning Journey

Welcome to my LangChain learning repository! This project showcases my exploration of the LangChain framework, covering key concepts, tools, and techniques for building AI-powered applications. The repo includes code examples, notes, and organized folders to demonstrate my understanding of LangChain's features.

## Overview

This repository is a collection of my hands-on learning with LangChain, a powerful framework for building applications with large language models (LLMs). Below is a summary of the topics covered, each organized into dedicated folders with code examples and notes.

### Repository Structure

- **chains/**: Examples of basic LangChain chains for sequential processing.
  - `example_chain.py`: Sample code for chaining prompts and models.
  - `notes.md`: Key takeaways and explanations.
- **chatmodels/**: Experiments with chat-based language models in LangChain.
- **embeddings/**: Code for generating and using embeddings for text representation.
- **generative-ai/**: Examples of generative AI tasks using LangChain.
- **agents/**: Implementations of LangChain agents for dynamic task handling.
  - **built-in-tools/**: Using LangChain's built-in tools.
  - **custom-tools/**: Custom tool creation for specific tasks.
- **tool-calling/**: Examples of invoking tools within LangChain workflows.
- **rag/**: Retrieval-Augmented Generation (RAG) implementations.
  - **splitters/**: Text splitting techniques for document processing.
  - **loaders/**: Document loaders for various data sources.
  - **vectorstores/**: Vector store setups for efficient retrieval.
  - **retrievers/**: Retriever configurations for RAG pipelines.
- **output-parsers/**: Parsing model outputs into structured formats.
  - **pydantic/**: Using Pydantic for structured output parsing.
  - **json/**: JSON-based output parsing examples.
  - **str/**: String-based output parsing.
- **structured-output/**: Additional examples of structured output handling.
- **prompts/**: Prompt engineering with LangChain.
  - **chatprompttemplate/**: Templates for chat-based interactions.
  - **prompttemplate/**: Standard prompt templates.
  - **messages/**: Managing message formats for conversations.
- **runnables/**: Examples of LangChain runnables for flexible workflows.
  - **branch/**: Conditional branching in workflows.
  - **parallel/**: Parallel processing examples.
  - **sequence/**: Sequential runnable chains.
  - **passthrough/**: Passthrough runnables for data flow.
  - **lambda/**: Custom lambda functions in runnables.

## Getting Started

To explore or run the examples in this repository, follow these steps:

### Prerequisites

- Python 3.8+
- Required libraries: `langchain`, `openai` (or other LLM provider), and dependencies specific to examples (e.g., `faiss-cpu` for vector stores, `pydantic` for output parsing).

Install dependencies:

```bash
pip install langchain openai faiss-cpu pydantic
```

> Note: Some examples may require additional packages. Check individual folders for specific requirements.

### Running Examples

1. Clone this repository:

   ```bash
   git clone https://github.com/NayyabMalik/My-learning-journey-with-LangChainNayyabMalik/My-learning-journey-with-LangChain
   cd My-learning-journey-with-LangChainMy-learning-journey-with-LangChain
   ```
2. Navigate to a specific folder (e.g., `chains/`) and run the example:

   ```bash
   python chains/example_chain.py
   ```
3. Check `notes.md` in each folder for context and explanations.

### Environment Setup

Some examples require an API key for an LLM provider (e.g., OpenAI). Set it up in your environment:

```bash
export OPENAI_API_KEY='your-api-key'
```

Replace `your-api-key` with your actual key.

## Purpose

This repository serves as a portfolio of my LangChain learning journey, created to solidify my understanding and demonstrate practical skills for AI development roles. It covers core concepts like chains, agents, and RAG, as well as advanced features like runnables and structured outputs.

## Future Plans

- Add more complex RAG applications with real-world datasets.
- Experiment with additional LangChain integrations (e.g., Hugging Face models).
- Include deployment examples (e.g., Streamlit apps).

## Contact

Feel free to reach out via GitHub Issues or nayyabm16@gmail.com for questions or feedback.

---

*Built with curiosity and a passion for AI-powered development!*