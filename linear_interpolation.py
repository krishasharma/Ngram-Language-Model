import math

def interpolated_probability(word, context, unigram_model, bigram_model, trigram_model, lambdas):
    lambda1, lambda2, lambda3 = lambdas
    unigram_prob = unigram_model.get(word, 1e-12)
    bigram_prob = bigram_model.get((context[-1],), {}).get(word, 1e-12) if len(context) >= 1 else unigram_prob
    trigram_prob = trigram_model.get(tuple(context[-2:]), {}).get(word, 1e-12) if len(context) >= 2 else bigram_prob
    
    return lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob


def interpolated_perplexity(data, unigram_model, bigram_model, trigram_model, lambdas):
    total_log_prob = 0.0
    N = 0  # Total number of words processed
    
    for line in data:
        # Assume `line` is already a list of words, and we just add START and STOP tokens
        words = ['<START>', '<START>'] + line + ['<STOP>']
        for i in range(2, len(words)):
            context = words[i-2:i]
            word = words[i]
            probability = interpolated_probability(word, context, unigram_model, bigram_model, trigram_model, lambdas)
            total_log_prob += math.log(probability)
            N += 1

    return math.exp(-total_log_prob / N) if N > 0 else float('inf')
