from collections import defaultdict, Counter
from data_processing import *
import math

def build_ngram_model(data, n, alpha=0):
    model = Counter()
    context_counts = Counter()
    
    for sentence in data:
        if n > 1:
            sentence = ['<START>'] * (n - 1) + sentence
        
        for i in range(len(sentence) - (n-1)):
            ngram = tuple(sentence[i:i+n])
            model[ngram] += 1
            if n > 1:
                context = tuple(sentence[i:i+n-1])
                context_counts[context] += 1

    if alpha > 0:
        vocab_size = len(set(word for sentence in data for word in sentence)) + 1
        for ngram in model.keys():
            model[ngram] += alpha
        for context in context_counts.keys():
            context_counts[context] += alpha * vocab_size

    return model, context_counts

def calculate_perplexity(model, context_counts, data, n, alpha=0):
    total_log_prob = 0
    total_tokens = 0

    for sentence in data:
        if n > 1:
            sentence = ['<START>'] * (n-1) + sentence
        
        for i in range(len(sentence) - (n-1)):
            ngram = tuple(sentence[i:i+n])
            context = tuple(sentence[i:i+n-1]) if n > 1 else None

            if context:
                if context in context_counts and ngram in model:
                    prob = model[ngram] / context_counts[context] if alpha > 0 else model[ngram] / context_counts[context]
                else:
                    prob = 1e-7
            else:
                total_count = sum(model.values())
                prob = model[ngram] / total_count if ngram in model else 1e-7

            total_log_prob += math.log(prob)
        
        total_tokens += len(sentence)

    perplexity = math.exp(-total_log_prob / total_tokens)
    return perplexity
