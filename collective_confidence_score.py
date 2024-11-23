# define function all the confidence scores 
# 1. logit 
# 2. entropy 
# 3. semantic simlarity (BERTScore)
# 4. perplexity
# 5. jenson shannon divergence (jsd) 


import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def calculate_logit_confidence(logits):

    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    return np.max(probabilities)  # Maximum probability


def calculate_entropy_confidence(logits):

    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    return -entropy(probabilities, base=2)  # Entropy in bits


def calculate_perplexity_confidence(logits):

    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    perplexity = np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))  # Perplexity
    return perplexity

def calculate_jsd_confidence(logits):

    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    uniform_prob = np.ones_like(probabilities) / len(probabilities)
    jsd = jensenshannon(probabilities, uniform_prob)  # Jensen-Shannon Divergence
    return jsd

