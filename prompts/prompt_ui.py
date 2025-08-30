from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Set up Hugging Face API token (ensure it's set in your environment or Streamlit secrets)
# Example: export HUGGINGFACEHUB_API_TOKEN='your_token_here'
# Or in Streamlit secrets (secrets.toml): HUGGINGFACEHUB_API_TOKEN = "your_token_here"

try:
    # Initialize the Hugging Face language model
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Fallback model; replace with deepseek-ai/DeepSeek-R1-0528 if confirmed available
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    chat_model = ChatHuggingFace(llm=llm)

    # Streamlit UI
    st.header("Research Tool")

    # Input widgets
    paper_input = st.selectbox(
        "Select Research Paper Name",
        [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-3: Language Models are Few-Shot Learners",
            "Diffusion Models Beat GANs on Image Synthesis",
        ],
    )

    style_input = st.selectbox(
        "Select Explanation Style",
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
    )

    length_input = st.selectbox(
        "Select Explanation Length",
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
    )

    # Load prompt template from template.json
    try:
        template = load_prompt("template.json")
    except Exception as e:
        st.error(f"Failed to load template.json: {str(e)}")
        st.stop()

    # Create LangChain chain
    try:
        chain = template | chat_model
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        st.stop()

    # Summarize button
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            try:
                # Invoke the chain with input dictionary
                result = chain.invoke(
                    {
                        "paper_input": paper_input,
                        "style_input": style_input,
                        "length_input": length_input,
                    }
                )
                # Display the result
                st.write(result.content)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

except Exception as e:
    st.error(f"Error initializing model: {str(e)}")