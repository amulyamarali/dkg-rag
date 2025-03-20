import streamlit as st
from langchain_openai import OpenAI
import agent1 as a1

# Streamlit setup
st.set_page_config(page_title="Smart Q&A System", page_icon="🤖", layout="centered")

# Define pastel color palette for styling
PRIMARY_COLOR = "#A8DADC"
SECONDARY_COLOR = "#457B9D"
BACKGROUND_COLOR = "#F1FAEE"
TEXT_COLOR = "#1D3557"

# CSS styling
st.markdown(
    f"""
    <style>
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Arial', sans-serif;
    }}
    .stButton button {{
        background-color: {PRIMARY_COLOR};
        color: {TEXT_COLOR};
        border: 1px solid {SECONDARY_COLOR};
        border-radius: 8px;
        padding: 0.5em 1em;
    }}
    .stTextInput div {{
        border-radius: 8px;
        border: 1px solid {SECONDARY_COLOR};
    }}
    .stTextInput div input {{
        padding: 0.5em;
        font-size: 1em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to initialize the local LLM
def load_local_llm():
    return OpenAI(
        base_url="http://localhost:1233/v1",  # Adjust to match your local server endpoint
        openai_api_key="dummy_key",  # Placeholder to bypass the check
        max_tokens=30,  # Limit to ensure short answers
    )

# Streamlit app layout
st.title("🌟 DKG-RAG")
st.subheader("Ask a question and get confident answers!")

st.markdown(
    """
    Welcome to the **Smart Q&A System**! Simply type your question below, and our system will generate an accurate response with confidence scores.
    """
)

# Input field for user question
question = st.text_input("Type your question here:", placeholder="E.g., What is the capital of France?")

if question:
    with st.spinner("Processing your question..."):
        # Initialize the local LLM
        llm = load_local_llm()

        # Generate answer using agent1
        answer, ccs, flag, triples = a1.generate_answer_a1(question, llm)

    # Display results
    st.markdown("### 💡 Final Answer")
    st.markdown(
        f"<div style='padding: 1em; background-color: {PRIMARY_COLOR}; border-radius: 8px;'><strong>{answer}</strong></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Weigted Collective Confidence Score")
    st.metric(label="Score", value=f"{ccs:.2f}")

    st.markdown("### ❌ or ✅ ?")
    if not flag:
        st.error("The response needs additional verification! ❌")
    else:
        st.success("The response is verified and complete! ✅")

    # Display extracted triples
    if triples:
        st.markdown("### 🔗 Extracted Triples")
        st.markdown("These are the extracted triples based on your question:")
        
        # Iterate over triples and display them
        for triple, description in triples:
            st.markdown(
                f"""
                <div style='padding: 0.5em; background-color: {SECONDARY_COLOR}; color: white; border-radius: 8px; margin-bottom: 0.5em;'>
                    <strong>Head:</strong> {triple[0]}<br>
                    <strong>Relation:</strong> {triple[1]}<br>
                    <strong>Tail:</strong> {triple[2]}<br>
                    <strong>Description:</strong> {description}
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown("---")
st.markdown(
    "Designed with ❤️ using [Streamlit](https://streamlit.io)."
)
