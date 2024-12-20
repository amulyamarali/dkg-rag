import evaluator as eval
import agent2 as a2
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import torch

def generate_knowledge_graph(file):
    try:
        df = pd.read_csv(file)
        result = []
        for _, row in df.iterrows():
            # Convert any non-string values to strings and handle NaN values
            sentence = str(row['Sentence']) if pd.notna(row['Sentence']) else ""
            head = str(row['Head']) if pd.notna(row['Head']) else ""
            relation = str(row['Relation']) if pd.notna(row['Relation']) else ""
            tail = str(row['Tail']) if pd.notna(row['Tail']) else ""
            
            if sentence and head and relation and tail:  # Only add if all fields are non-empty
                result.append((sentence, (head, relation, tail)))
        return result
    except Exception as e:
        print(f"Error in generate_knowledge_graph: {e}")
        return []

def find_top_relevant_triples(question, sentence_triples, top_n=3):
    try:
        question_words = set(question.lower().split())
        triple_scores = []

        for sentence, triple in sentence_triples:
            try:
                # Ensure we're working with strings
                head = str(triple[0])
                tail = str(triple[2])
                
                head_tail_words = set(head.lower().split()) | set(tail.lower().split())
                common_words = question_words & head_tail_words
                score = len(common_words)
                if score > 0:
                    triple_scores.append((triple, score, sentence))
            except Exception as e:
                print(f"Error processing triple: {e}")
                continue

        top_triples = sorted(triple_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [(triple[0], triple[2]) for triple in top_triples]
    except Exception as e:
        print(f"Error in find_top_relevant_triples: {e}")
        return []

def calculate_confidence(question, sentence, answer):
    try:
        question_words = set(question.lower().split())
        sentence_words = set(sentence.lower().split())
        answer_words = set(answer.lower().split())

        question_overlap = question_words & sentence_words
        answer_overlap = answer_words & sentence_words

        denominator = len(question_words) + len(sentence_words) + len(answer_words)
        if denominator == 0:
            return 0.0
            
        confidence = (len(question_overlap) + len(answer_overlap)) / denominator
        return confidence
    except Exception as e:
        print(f"Error in calculate_confidence: {e}")
        return 0.0

def get_answer_from_llm(question, sentence, llm, threshold=0.5):
    try:
        prompt = f"Context: {sentence}\n\nQuestion: {question}\nAnswer:"
        response = llm.generate([prompt])
        
        answer = response.generations[0][0].text.strip()
        confidence_score = calculate_confidence(question, sentence, answer)

        return answer, confidence_score
    except Exception as e:
        print(f"Error in get_answer_from_llm: {e}")
        return "I couldn't generate an answer.", 0.0

def answer_question(question, sentence_triples, llm, threshold=0.5):
    try:
        top_relevant_triples = find_top_relevant_triples(question, sentence_triples)
        if not top_relevant_triples:
            return "I couldn't find relevant information.", []

        unique_sentences = list(set([triple[1] for triple in top_relevant_triples]))
        combined_context = " ".join(unique_sentences)
        
        answer, confidence_score = get_answer_from_llm(question, combined_context, llm, threshold)
        return answer, top_relevant_triples
    except Exception as e:
        print(f"Error in answer_question: {e}")
        return "An error occurred while processing the question.", []

def generate_answer_a1(question, llm):
    try:
        file = 'relations_groq.csv'
        sentence_triples = generate_knowledge_graph(file)
        
        if not sentence_triples:
            return "Could not process knowledge graph.", 0.0, False, []

        response, triples = answer_question(question, sentence_triples, llm)

        # Load model and tokenizer
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        inputs = tokenizer(question, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        max_entropy = np.log(len(tokenizer))
        flag, ccs = eval.calculate_metrics(outputs, max_entropy)

        print(f"########################### CCS: {ccs}")

        if not flag:
            response = a2.generate_answer_a2(question, sentence_triples, llm, tokenizer)

        return response, ccs, flag, triples
    except Exception as e:
        print(f"Error in generate_answer_a1: {e}")
        return "An error occurred.", 0.0, False, []