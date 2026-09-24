import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_vocabulary(documents):
    words = sorted(set(word for doc in documents for word in doc.split()))
    return {word: i for i, word in enumerate(words)}


def compute_counts(documents, vocabulary):
    X = np.zeros((len(documents), len(vocabulary)))

    for i, doc in enumerate(documents):
        counts = Counter(doc.split())

        for word, count in counts.items():
            if word in vocabulary:
                X[i, vocabulary[word]] = count

    return X


def compute_tf(documents, vocabulary):
    counts = compute_counts(documents, vocabulary)
    tf = np.zeros_like(counts, dtype=float)

    for i, doc in enumerate(documents):
        n = len(doc.split())
        if n > 0:
            tf[i] = counts[i] / n

    return tf


def compute_idf(documents, vocabulary):
    N = len(documents)
    df = np.zeros(len(vocabulary))

    for doc in documents:
        for word in set(doc.split()):
            if word in vocabulary:
                df[vocabulary[word]] += 1

    idf = np.zeros(len(vocabulary))
    mask = df > 0
    idf[mask] = np.log(N / df[mask])

    return idf


def compute_tfidf(documents, vocabulary):
    return compute_tf(documents, vocabulary) * compute_idf(documents, vocabulary)


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return 0.0

    return float(np.dot(x, y) / denominator)


def run_tests():
    docs = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    vocab = build_vocabulary(docs)

    assert vocab == {
        "cat": 0,
        "dog": 1,
        "eats": 2,
        "fish": 3,
        "likes": 4
    }

    counts = compute_counts(docs, vocab)
    assert np.array_equal(
        counts,
        np.array([
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1]
        ])
    )

    tf = compute_tf(docs, vocab)
    assert np.isclose(tf[0, vocab["cat"]], 1 / 3)

    idf = compute_idf(docs, vocab)
    assert np.isclose(idf[vocab["cat"]], np.log(3 / 2))
    assert np.isclose(idf[vocab["fish"]], 0.0)

    tfidf = compute_tfidf(docs, vocab)
    assert np.isclose(
        tfidf[0, vocab["cat"]],
        (1 / 3) * np.log(3 / 2)
    )

    assert np.isclose(
        cosine_similarity([1, 1, 1], [1, 1, 0]),
        2 / np.sqrt(6)
    )

    print("All tests passed.")


def compare_with_reference():
    docs = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    vocab = build_vocabulary(docs)

    student_counts = compute_counts(docs, vocab)
    student_tf = compute_tf(docs, vocab)
    student_idf = compute_idf(docs, vocab)
    student_tfidf = compute_tfidf(docs, vocab)

    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False)
    reference_counts = vectorizer.fit_transform(docs).toarray().astype(float)

    reference_tf = reference_counts / reference_counts.sum(axis=1, keepdims=True)

    transformer = TfidfTransformer(
        norm=None,
        use_idf=True,
        smooth_idf=False
    )
    transformer.fit(reference_counts)

    reference_idf = transformer.idf_ - 1
    reference_tfidf = reference_tf * reference_idf

    student_cos = cosine_similarity([1, 1, 1], [1, 1, 0])
    reference_cos = sklearn_cosine(
        np.array([[1, 1, 1]]),
        np.array([[1, 1, 0]])
    )[0, 0]

    print("Counts:", np.allclose(student_counts, reference_counts))
    print("TF:", np.allclose(student_tf, reference_tf))
    print("IDF:", np.allclose(student_idf, reference_idf))
    print("TF-IDF:", np.allclose(student_tfidf, reference_tfidf))
    print("Cosine:", np.isclose(student_cos, reference_cos))


if __name__ == "__main__":
    run_tests()
    compare_with_reference()
