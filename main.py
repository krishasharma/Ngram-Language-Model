from ngram_model import *
from data_processing import *
from linear_interpolation import *

def main():
    train_data = read_data('A2-Data/1b_benchmark.train.tokens')
    dev_data = read_data('A2-Data/1b_benchmark.dev.tokens')
    test_data = read_data('A2-Data/1b_benchmark.test.tokens')

    preprocessed_train, train_token_counts = preprocess(train_data)
    preprocessed_dev, _ = preprocess(dev_data)
    preprocessed_test, _ = preprocess(test_data)

    # Build models
    unigram_model, unigram_context_counts = build_ngram_model(preprocessed_train, 1)
    bigram_model, bigram_context_counts = build_ngram_model(preprocessed_train, 2)
    trigram_model, trigram_context_counts = build_ngram_model(preprocessed_train, 3)
    
   # Calculate and report perplexity
    print("Unigram Perplexity on Test Set:", calculate_perplexity(unigram_model, unigram_context_counts, preprocessed_test, 1))
    print("Bigram Perplexity on Test Set:", calculate_perplexity(bigram_model, bigram_context_counts, preprocessed_test, 2))
    print("Trigram Perplexity on Test Set:", calculate_perplexity(trigram_model, trigram_context_counts, preprocessed_test, 3))

    alpha_values = [0.1, 0.5, 1]

    for alpha in alpha_values:
        # Train smoothed models
        smoothed_unigram_model, smoothed_context_counts = build_ngram_model(preprocessed_train, 1, alpha)
        smoothed_bigram_model, smoothed_context_counts1 = build_ngram_model(preprocessed_train, 2, alpha)
        smoothed_trigram_model, smoothed_context_counts2 = build_ngram_model(preprocessed_train, 3, alpha)

        # Calculate perplexity for each model on the development set
        train_perplexity_unigram = calculate_perplexity(unigram_model, smoothed_context_counts, preprocessed_test, 1, alpha)
        dev_perplexity_bigram = calculate_perplexity(bigram_model, smoothed_context_counts1, preprocessed_test, 2, alpha)
        test_perplexity_trigram = calculate_perplexity(trigram_model, smoothed_context_counts2, preprocessed_test, 3, alpha)

        print(f"Perplexity for Alpha {alpha}:")
        print("Unigram:", train_perplexity_unigram)
        print("Bigram:", dev_perplexity_bigram)
        print("Trigram:", test_perplexity_trigram)

    lambdas = (0.1, 0.3, 0.6)

    test_perplexity = interpolated_perplexity(preprocessed_test, unigram_model, bigram_model, trigram_model, lambdas)
    print(f"Interpolated Perplexity: {test_perplexity}")
    

if __name__ == "__main__":
    main()