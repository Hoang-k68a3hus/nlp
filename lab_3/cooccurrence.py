import numpy as np


def build_vocabulary(corpus):
    vocab = set()

    for sentence in corpus:
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence

        for word in words:
            vocab.add(word)

    return sorted(vocab)


def build_cooccurrence_matrix(corpus, vocabulary, window=1):
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)

    for sentence in corpus:
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence

        for i, word in enumerate(words):
            if word not in word_to_idx:
                continue

            start = max(0, i - window)
            end = min(len(words), i + window + 1)

            for j in range(start, end):
                if i == j:
                    continue

                context_word = words[j]

                if context_word in word_to_idx:
                    row = word_to_idx[word]
                    col = word_to_idx[context_word]
                    matrix[row][col] += 1

    return matrix


def cosine_similarity(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return 0.0

    return float(np.dot(x, y) / denominator)


def most_similar(word, matrix, vocabulary, top_k=5):
    if word not in vocabulary:
        raise ValueError("Word not found in vocabulary")

    word_index = vocabulary.index(word)
    word_vector = matrix[word_index]
    scores = []

    for i, other_word in enumerate(vocabulary):
        if other_word == word:
            continue

        score = cosine_similarity(word_vector, matrix[i])
        scores.append((other_word, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]
