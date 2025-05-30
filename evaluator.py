import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import collective_confidence_score as wccs


def calculate_normalized_metrics(logit_conf, entropy_conf, perplexity, jsd, max_entropy):
    weights = [0.2, 0.25, 0.2, 0.35]

    # Normalize metrics
    normalized_entropy = 1 - (entropy_conf / max_entropy)
    normalized_perplexity = 1 / perplexity  # Lower perplexity is better, hence inverse
    normalized_jsd = 1 - jsd  # Lower JSD is better, hence inverse

    # CCS calculation using weighted average
    wccs = (weights[0] * logit_conf +
           weights[1] * normalized_entropy +
           weights[2] * normalized_perplexity +
           weights[3] * normalized_jsd)

    return wccs

def calculate_metrics(outputs, max_entropy):
    # Aggregate logits
    start_logits = outputs.start_logits[0].numpy()
    end_logits = outputs.end_logits[0].numpy()
    combined_logits = start_logits + end_logits

    # Calculate metrics
    logit_conf = wccs.calculate_logit_confidence(combined_logits)
    entropy_conf = wccs.calculate_entropy_confidence(combined_logits)
    jsd = wccs.calculate_jsd_confidence(combined_logits)
    perplexity = wccs.calculate_perplexity_confidence(combined_logits)

    ccs = calculate_normalized_metrics(logit_conf, entropy_conf, perplexity, jsd, max_entropy)

    if ccs > 0.6:
        return True, ccs

    else:
        return False, ccs





