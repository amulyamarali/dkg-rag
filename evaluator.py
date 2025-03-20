import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import collective_confidence_score as wccs


# def calculate_normalized_metrics(logit_conf, entropy_conf, perplexity, jsd, max_entropy):
#     weights = [0.4, 0.25, 0.2, 0.15]

#     # Normalize metrics
#     normalized_entropy = 1 - (entropy_conf / max_entropy)
#     normalized_perplexity = 1 / perplexity  # Lower perplexity is better, hence inverse
#     normalized_jsd = 1 - jsd  # Lower JSD is better, hence inverse

#     # CCS calculation using weighted average
#     wccs = (weights[0] * logit_conf +
#            weights[1] * normalized_entropy +
#            weights[2] * normalized_perplexity +
#            weights[3] * normalized_jsd)

#     return wccs

def calculate_normalized_metrics(logit_conf, entropy_conf, perplexity, jsd, max_entropy, max_cos_sim):
    weights = [0.4, 0.2, 0.3, 0.3, 0.2]
    
    # Normalize entropy by maximum entropy
    # max_entropy = np.log2(num_classes + 1)  # Maximum possible entropy for N classes
    entropy = (entropy_conf / max_entropy)  # Ensure within [0,1]

    normalized_entropy = 1 / entropy  # Lower entropy is better

    # Normalize perplexity using exponential decay
    normalized_perplexity = 1 - (1 / (perplexity))  # Scales into [0,1]

    # Normalize JSD (1 - JSD)
    normalized_jsd = 1 - jsd  # Lower JSD is better

    print("ORIGINAL METRICS: ", logit_conf, entropy_conf, perplexity, jsd)

    print("Logit Confidence: ", logit_conf)
    print("Normalized Entropy: ", normalized_entropy)
    print("Normalized Perplexity: ", normalized_perplexity)
    print("Normalized JSD: ", normalized_jsd)


    # CCS calculation using weighted sum
    # ccs = (weights[0] * logit_conf +
    #        weights[1] * normalized_entropy +
    #        weights[2] * normalized_perplexity +
    #        weights[3] * normalized_jsd)

    dummy = (weights[1] * normalized_entropy +
           (weights[2]+0.1) * normalized_perplexity +
           (weights[3]+0.1)* normalized_jsd)
    
    print("Dummy: ", dummy)

    
    ccs = (weights[1] * normalized_entropy +
           weights[2] * normalized_perplexity +
           weights[3] * normalized_jsd+
           weights[4] * max_cos_sim)
    


    return ccs

def calculate_metrics(outputs, max_entropy, max_cos_sim):
    # Aggregate logits
    # start_logits = outputs.start_logits[0].numpy()
    # end_logits = outputs.end_logits[0].numpy()

    start_logits = outputs["start_logits"][0].numpy()
    end_logits = outputs["end_logits"][0].numpy()

    combined_logits = start_logits + end_logits


    # print("Combined Logits: ", combined_logits)

    print("Length of combined logits: ", len(combined_logits))

    len_combined_logits = len(combined_logits)

    # Calculate metrics
    logit_conf = wccs.calculate_logit_confidence(combined_logits)
    entropy_conf = wccs.calculate_entropy_confidence(combined_logits)
    jsd = wccs.calculate_jsd_confidence(combined_logits)
    perplexity = wccs.calculate_perplexity_confidence(combined_logits)

    ccs = calculate_normalized_metrics(logit_conf, entropy_conf, perplexity, jsd, max_entropy, max_cos_sim)

    print("entropy_conf: ", entropy_conf)

    if ccs > 0.6:
        return True, ccs

    else:
        return False, ccs





