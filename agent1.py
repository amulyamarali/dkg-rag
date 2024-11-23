# Function in agent1.py
# 1. generate_answer 

import evaluator as eval
import agent2 as a2
import pandas as pd
import evaluator as eval
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import numpy as np
import torch



def generate_knowledge_graph(file):
    df = pd.read_csv(file)
    result = []
    sentence = df['Sentence']
    head = df['Head']
    relation = df['Relation']
    tail = df['Tail']

    for i in range(len(sentence)):
        result.append((sentence[i], (head[i], relation[i], tail[i])))
    return result



def find_top_relevant_triples(question, sentence_triples, top_n=3):
    question_words = set(question.lower().split())
    triple_scores = []

    for sentence, triple in sentence_triples:
        head_tail_words = set(triple[0].lower().split()) | set(triple[2].lower().split())
        common_words = question_words & head_tail_words
        score = len(common_words)
        if score > 0:
            triple_scores.append((triple, score, sentence))

    top_triples = sorted(triple_scores, key=lambda x: x[1], reverse=True)[:top_n]
    print('top_triples:', top_triples)
    return [(triple[0], triple[2]) for triple in top_triples]

# Function to compute a confidence score (fallback when LLM doesn't return scores)
def calculate_confidence(question, sentence, answer):
    question_words = set(question.lower().split())
    sentence_words = set(sentence.lower().split())
    answer_words = set(answer.lower().split())

    # Compute overlaps
    question_overlap = question_words & sentence_words
    answer_overlap = answer_words & sentence_words

    # Confidence is proportional to the overlaps
    confidence = (len(question_overlap) + len(answer_overlap)) / (len(question_words) + len(sentence_words) + len(answer_words))
    return confidence

# Function to get an answer from the local LLM based on question and sentence context
def get_answer_from_llm(question, sentence, llm, threshold=0.5):
    prompt = f"Context: {sentence}\n\nQuestion: {question}\nAnswer:"
    response = llm.generate([prompt])

    answer = response.generations[0][0].text.strip()
    confidence_score = calculate_confidence(question, sentence, answer)

    return answer, confidence_score

def answer_question(question, sentence_triples, llm, threshold=0.5):
    top_relevant_triples = find_top_relevant_triples(question, sentence_triples)

    print("Top relevant triples: ", top_relevant_triples)

    unique_sentences = list(set([triple[1] for triple in top_relevant_triples]))

    # Combine all unique sentences into a single context
    combined_context = " ".join(unique_sentences)

    print("Combined context: ", combined_context)

    print("Top 3 relevant triples: ")
    for triple in top_relevant_triples:
        print(triple)

    answer, confidence_score = get_answer_from_llm(question, combined_context , llm, threshold)


    print('ANSWER FROM LLM:', answer )
    print('Confidence:', confidence_score)

    return answer

def generate_answer_a1(question,llm):
    file = 'relations_2.csv'
    sentence_triples = generate_knowledge_graph(file)
    # print('sentence_triples:', sentence_triples)
    response = answer_question(question, sentence_triples, llm)

    print("Agent 1 response: ", response)   
    # Load model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(question, return_tensors="pt", padding=True)
    with torch.no_grad():
            outputs = model(**inputs)

    # print("OUTPUTS START LOGITS",outputs.start_logits[0].cpu().numpy())

    max_entropy = np.log(len(tokenizer))

    flag, ccs = eval.calculate_metrics(outputs, max_entropy)
    print('flag:', flag)
    print('ccs:', ccs)

    if not flag: # false
        response = a2.generate_answer_a2(question, sentence_triples,llm, tokenizer)
        print("Agent 2 response: ", response)

    return response



from langchain_openai import OpenAI


# Initialize local OpenAI LLM
def load_local_llm():
    return OpenAI(
        base_url="http://localhost:1233/v1",  # Adjust to match your local server endpoint
        openai_api_key="dummy_key",  # Placeholder to bypass the check
        max_tokens=20  # Limit to ensure short answers
    )

# Initialize the local LLM
llm = load_local_llm()
res = generate_answer_a1('how many times catholic mass celebrates per week ?', llm)

print(res)