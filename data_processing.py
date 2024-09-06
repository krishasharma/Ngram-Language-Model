from collections import Counter

def preprocess(data):
    token_counts = Counter()
    processed_data = []

    for line in data:
        words = line.strip().split()
        token_counts.update(words)

    for line in data:
        words = line.strip().split()
        processed_line = [word if token_counts[word] >= 3 else '<UNK>' for word in words]
        processed_line.append('<STOP>')  # Append <STOP> to each sentence
        processed_data.append(processed_line)

    return processed_data, token_counts

def read_data(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        data = file.readlines()
    return data