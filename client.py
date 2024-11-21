# calling agent1 funciton to generate answer

# calling fucntion to check for ccs from evaluator 

# based on ccs, calling agent2 function to complete answer by generating triples and displaying the final asnwer by the AGENT2 to the users on the streamlit app


import agent1 as a1
from langchain_openai import OpenAI



question = input()

# Initialize local OpenAI LLM
def load_local_llm():
    return OpenAI(
        base_url="http://localhost:1233/v1",  # Adjust to match your local server endpoint
        openai_api_key="dummy_key",  # Placeholder to bypass the check
        max_tokens=20  # Limit to ensure short answers
    )

# Initialize the local LLM
llm = load_local_llm()

answer = a1.generate_answer_a1(question,llm)