# Function in agent1.py
# 1. generate_answer 

import evaluator as eval
import agent2 as a2

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
    return [(triple[0], triple[2]) for triple in top_triples]

def generate_answer_a1(question,llm):

    # find triples and
    return 