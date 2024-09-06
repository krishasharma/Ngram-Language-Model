import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the directory exists
output_dir = 'asgn2'
os.makedirs(output_dir, exist_ok=True)

# Provided perplexity data
data = {
    'Model': ['Unigram', 'Bigram', 'Trigram'],
    'Test Set Perplexity': [652.3522149462672, 403.146484840783, 11733.912699818216]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the perplexity scores for n-gram models
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Test Set Perplexity'], color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Test Set Perplexity')
plt.title('Perplexity Scores for n-gram Models on Test Set')
plt.yscale('log')  # Using logarithmic scale for better visualization
plt.savefig(os.path.join(output_dir, 'ngram_perplexity.png'))
plt.show()

# Additive Smoothing perplexity data
additive_data = {
    'Alpha': [0.1, 0.5, 1.0],
    'Unigram': [652.3522149462672, 652.3522149462672, 652.3522149462672],
    'Bigram': [1002.7874288489946, 1922.6575800834044, 2735.7174787905983],
    'Trigram': [42647.58734160999, 74414.15324481901, 96506.08968790872]
}

# Create a DataFrame
additive_df = pd.DataFrame(additive_data)

# Plot the perplexity scores for additive smoothing
plt.figure(figsize=(10, 6))
for model in ['Unigram', 'Bigram', 'Trigram']:
    plt.plot(additive_df['Alpha'], additive_df[model], marker='o', label=model)
plt.xlabel('Alpha')
plt.ylabel('Perplexity')
plt.title('Perplexity Scores for Additive Smoothing')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'additive_smoothing_perplexity.png'))
plt.show()
