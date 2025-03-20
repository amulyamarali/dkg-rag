import evaluator as eval
import agent2 as a2
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer, util



def generate_knowledge_graph(file):
    try:
        df = pd.read_csv(file)
        result = []
        for _, row in df.iterrows():
            sentence = str(row['Sentence']) if pd.notna(row['Sentence']) else ""
            head = str(row['Head']) if pd.notna(row['Head']) else ""
            relation = str(row['Relation']) if pd.notna(row['Relation']) else ""
            tail = str(row['Tail']) if pd.notna(row['Tail']) else ""
            
            if sentence and head and relation and tail:
                result.append((sentence, (head, relation, tail)))
        return result
    except Exception as e:
        print(f"Error in generate_knowledge_graph: {e}")
        return []

def find_top_relevant_triples(question, sentence_triples, top_n=3):   ##### PLEASE FIND AN ALTERNATIVE FOR THIS FUNCTION
    try:
        question_words = set(question.lower().split())
        triple_scores = []

        for sentence, triple in sentence_triples:
            try:
                head = str(triple[0])
                tail = str(triple[2])
                
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",set(head.lower().split()))
                head_tail_words = set(head.lower().split()) | set(tail.lower().split())
                common_words = question_words & head_tail_words
                score = len(common_words)
                if score > 0:
                    triple_scores.append((triple, score, sentence))
            except Exception as e:
                print(f"Error processing triple: {e}")
                continue

        top_triples = sorted(triple_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [(triple[0], sentence) for triple, score, sentence in top_triples]
    except Exception as e:
        print(f"Error in find_top_relevant_triples: {e}")
        return []

# ALTERNATIVE 1 
def find_top_relevant_triples_jaccard(question, sentence_triples, top_n=3):
    try:
        question_words = set(question.lower().split())
        triple_scores = []

        for sentence, triple in sentence_triples:
            try:
                head = str(triple[0])
                tail = str(triple[2])

                head_tail_words = set(head.lower().split()) | set(tail.lower().split())
                intersection = question_words & head_tail_words
                union = question_words | head_tail_words
                
                score = len(intersection) / len(union) if union else 0
                if score > 0:
                    triple_scores.append((triple, score, sentence))
            except Exception as e:
                print(f"Error processing triple: {e}")
                continue

        top_triples = sorted(triple_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return [(triple[0], sentence) for triple, score, sentence in top_triples]
    except Exception as e:
        print(f"Error in find_top_relevant_triples_jaccard: {e}")
        return []

# ALTERNATIVE 2
def find_top_relevant_triples_ashwini(question, sentence_triples, top_n=3):
    try:
        # Load a pre-trained Sentence Transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the question
        question_embedding = model.encode(question, convert_to_tensor=True)
        
        triple_scores = []
        sim = []
        for sentence, triple in sentence_triples:
            try:
                new_sim = []
                # Convert triple into a single text representation
                head, relation, tail = str(triple[0]), str(triple[1]), str(triple[2])
                triple_text = f"{head} {relation} {tail}"

                head_emb = model.encode(head, convert_to_tensor=True)
                relation_emb = model.encode(relation, convert_to_tensor=True)
                tail_emb = model.encode(tail, convert_to_tensor=True)

                sim_head = util.pytorch_cos_sim(question_embedding, head_emb).item()
                sim_relation = util.pytorch_cos_sim(question_embedding, relation_emb).item()
                sim_tail = util.pytorch_cos_sim(question_embedding, tail_emb).item()

            

                # Compute embeddings and similarity
                triple_embedding = model.encode(triple_text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(question_embedding, triple_embedding).item()

                sim.append((similarity, sentence, triple, sim_head, sim_relation, sim_tail))

                new_sim.append(similarity)
                new_sim.append(sim_head)
                new_sim.append(sim_relation)
                new_sim.append(sim_tail)

                new_sim.sort(reverse=True)
                if (new_sim[0]+new_sim[1]) > 0.7:
                    triple_scores.append((triple, similarity, sentence))
            
            except Exception as e:
                print(f"Error processing triple: {e}")
                continue
        
        sim = sorted(sim, key=lambda x: x[0], reverse=True)
        for i in range(3):
            print("************************************",sim[i])
        # Sort by similarity score in descending order
        top_triples = sorted(triple_scores, key=lambda x: x[1], reverse=True)[:top_n]

        return [(triple, sentence) for triple, score, sentence in top_triples]

    except Exception as e:
        print(f"Error in find_top_relevant_triples_ashwini: {e}")
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
        prompt = f"Context: {sentence}\n\n Make sure the answer is crisp and takes input only from the provided context\n\n Question: {question}\nAnswer:"
        response = llm.generate([prompt])
        
        answer = response.generations[0][0].text.strip()
        confidence_score = calculate_confidence(question, sentence, answer)

        return answer, confidence_score
    except Exception as e:
        print(f"Error in get_answer_from_llm: {e}")
        return "I couldn't generate an answer.", 0.0

def answer_question(question, sentence_triples, llm, threshold=0.5):
    try:
        # top_relevant_triples = find_top_relevant_triples(question, sentence_triples)
        top_relevant_triples = find_top_relevant_triples_ashwini(question, sentence_triples)

        print("TOP RELEVENT TRIPLES",top_relevant_triples)

        if not top_relevant_triples:
            return "I couldn't find relevant information.", [], 0.0

        unique_sentences = list(set([triple[1] for triple in top_relevant_triples]))
        combined_context = " ".join(unique_sentences)
        
        answer, confidence_score = get_answer_from_llm(question, combined_context, llm, threshold)

        # do cosine sim between answer and top_relevant_triples
        model = SentenceTransformer('all-MiniLM-L6-v2')
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        cos_sim = []
        for sentence in unique_sentences:
            embedding = model.encode(sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(answer_embedding, embedding).item()
            cos_sim.append(similarity)
        
        print("COSINE SIMILARITY: ", cos_sim)

        return answer, top_relevant_triples, confidence_score, max(cos_sim)
    except Exception as e:
        print(f"Error in answer_question: {e}")
        return "An error occurred while processing the question.", [], 0.0

import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def generate_answer_a1(question, llm):
    try:
        file = 'relations_groq_squad.csv'
        sentence_triples = generate_knowledge_graph(file)

        if not sentence_triples:
            return "Could not process knowledge graph.", 0.0, False, []

        # Generate response using knowledge graph
        response, triples, confidence_score, max_cos_sim = answer_question(question, sentence_triples, llm)
        print("########################### Response: ", response)

        # Load the tokenizer and model
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize question and response
        question_inputs = tokenizer(question, return_tensors="pt", padding=True)
        response_inputs = tokenizer(response, return_tensors="pt", padding=True)

        # Simulate QA model output for response (dummy logits for a similar format)
        response_length = response_inputs['input_ids'].shape[1]
        start_logits = torch.randn(1, response_length)  # Simulated start logit tensor
        end_logits = torch.randn(1, response_length)    # Simulated end logit tensor

        # Create an output structure similar to QuestionAnsweringModelOutput
        response_tensor = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": None,
            "attentions": None
        }

        # Compute QA model outputs for question
        with torch.no_grad():
            outputs = model(**question_inputs)

        print("########################### Outputs: ", outputs)
        print("########################### Response Tensor: ", response_tensor)

        # Compute evaluation metrics using response
        max_entropy = np.log(len(question.split()) + 1)

        print("question count: ", len(question.split())+1)

        print("########################### Max Entropy: ", max_entropy)
        flag, ccs = eval.calculate_metrics(response_tensor, max_entropy, max_cos_sim)

        print(f"########################### CCS: {ccs}")
        print("FLAG: ", flag)

        # If flag is False, generate an alternative response
        if not flag:
            response = a2.generate_answer_a2(question, sentence_triples, llm)

        return response, ccs, flag, triples

    except Exception as e:
        print(f"Error in generate_answer_a1: {e}")
        return "An error occurred.", 0.0, False, []
