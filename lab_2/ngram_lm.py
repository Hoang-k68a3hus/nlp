"""ngram_lm.py

Simple n-gram language model implementation for LAB 02.
Implements unigram, bigram, trigram, MLE/Laplace smoothing,
sentence probability, log probability, perplexity and next-word prediction.

No language-model library is used; counts are built directly with Counter.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple, Union


Sentence = Union[str, Sequence[str]]
Corpus = Iterable[Sentence]


def tokenize(sentence: Sentence) -> List[str]:
    """Convert a sentence into a list of tokens.

    - str -> split by whitespace
    - list/tuple of tokens -> copied to a list
    """
    if isinstance(sentence, str):
        return sentence.strip().split()
    return list(sentence)


def _prepare_corpus(corpus: Corpus) -> List[List[str]]:
    """Normalize corpus to List[List[str]] without crossing sentence boundaries."""
    return [tokenize(sentence) for sentence in corpus]


def build_vocabulary(corpus: Corpus) -> List[str]:
    """Return the sorted vocabulary of the corpus."""
    sentences = _prepare_corpus(corpus)
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    return sorted(vocab)


def count_ngrams(corpus: Corpus, n: int) -> Counter:
    """Count all n-grams in a corpus.

    Each n-gram is stored as a tuple, for example:
        ('the', 'cat')
        ('the', 'cat', 'eats')
    N-grams never cross sentence boundaries.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    sentences = _prepare_corpus(corpus)
    counts = Counter()

    for tokens in sentences:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            counts[ngram] += 1

    return counts


def train_unigram(corpus: Corpus) -> Counter:
    """Return unigram counts."""
    return count_ngrams(corpus, 1)


def train_bigram(corpus: Corpus) -> Counter:
    """Return bigram counts."""
    return count_ngrams(corpus, 2)


def train_trigram(corpus: Corpus) -> Counter:
    """Return trigram counts."""
    return count_ngrams(corpus, 3)


class NGramLanguageModel:
    """A simple unigram/bigram/trigram language model.

    Parameters
    ----------
    n : int
        1 = unigram, 2 = bigram, 3 = trigram.
    smoothing : str or None
        None / 'mle' for maximum-likelihood estimation.
        'laplace' for add-one smoothing.
    """

    def __init__(self, n: int = 2, smoothing: str = None):
        if n not in (1, 2, 3):
            raise ValueError("This lab implementation supports n = 1, 2, or 3")

        if smoothing is not None:
            smoothing = smoothing.lower()
        if smoothing not in (None, "mle", "laplace"):
            raise ValueError("smoothing must be None, 'mle', or 'laplace'")

        self.n = n
        self.smoothing = smoothing

        self.vocabulary = set()
        self.vocab_size = 0
        self.total_tokens = 0

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.is_fitted = False

    def fit(self, corpus: Corpus):
        """Train the model from a corpus.

        The model keeps counts up to its requested order so trigram sentence
        scoring can still score the first token with unigram probability and
        the second token with bigram probability.
        """
        sentences = _prepare_corpus(corpus)

        self.vocabulary = set(build_vocabulary(sentences))
        self.vocab_size = len(self.vocabulary)
        self.total_tokens = sum(len(sentence) for sentence in sentences)

        self.unigram_counts = train_unigram(sentences)
        self.bigram_counts = train_bigram(sentences) if self.n >= 2 else Counter()
        self.trigram_counts = train_trigram(sentences) if self.n >= 3 else Counter()

        self.is_fitted = True
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit(corpus) first.")

    def _use_laplace(self) -> bool:
        return self.smoothing == "laplace"

    def _unigram_probability(self, word: str) -> float:
        """P(word)."""
        # The model uses a closed training vocabulary. Laplace smoothing
        # handles unseen n-grams made of known words, not arbitrary OOV words.
        if word not in self.vocabulary:
            return 0.0

        count = self.unigram_counts[(word,)]

        if self._use_laplace():
            if self.vocab_size == 0:
                return 0.0
            return (count + 1) / (self.total_tokens + self.vocab_size)

        if self.total_tokens == 0:
            return 0.0
        return count / self.total_tokens

    def _bigram_probability(self, previous_word: str, word: str) -> float:
        """P(word | previous_word)."""
        if word not in self.vocabulary:
            return 0.0

        bigram_count = self.bigram_counts[(previous_word, word)]
        context_count = self.unigram_counts[(previous_word,)]

        if self._use_laplace():
            if self.vocab_size == 0:
                return 0.0
            return (bigram_count + 1) / (context_count + self.vocab_size)

        if context_count == 0:
            return 0.0
        return bigram_count / context_count

    def _trigram_probability(self, word1: str, word2: str, word: str) -> float:
        """P(word | word1, word2)."""
        if word not in self.vocabulary:
            return 0.0

        trigram_count = self.trigram_counts[(word1, word2, word)]
        context_count = self.bigram_counts[(word1, word2)]

        if self._use_laplace():
            if self.vocab_size == 0:
                return 0.0
            return (trigram_count + 1) / (context_count + self.vocab_size)

        if context_count == 0:
            return 0.0
        return trigram_count / context_count

    def probability(self, context: Sentence, word: str) -> float:
        """Return P(word | context).

        The last n-1 context words are used.

        For the beginning of a sentence, where fewer context words exist:
        - use unigram for the first word;
        - use bigram for the second word in a trigram model.
        """
        self._check_fitted()
        context_tokens = tokenize(context)

        if self.n == 1 or len(context_tokens) == 0:
            return self._unigram_probability(word)

        if self.n == 2 or len(context_tokens) == 1:
            return self._bigram_probability(context_tokens[-1], word)

        return self._trigram_probability(
            context_tokens[-2], context_tokens[-1], word
        )

    def sentence_probability(self, sentence: Sentence) -> float:
        """Return the probability of a sentence.

        This direct product is useful for short sentences. For long sentences,
        sentence_log_probability() is numerically safer.
        """
        self._check_fitted()
        tokens = tokenize(sentence)

        if not tokens:
            return 1.0

        probability_value = 1.0

        for i, word in enumerate(tokens):
            start = max(0, i - (self.n - 1))
            context = tokens[start:i]
            p = self.probability(context, word)

            if p == 0.0:
                return 0.0

            probability_value *= p

        return probability_value

    def sentence_log_probability(self, sentence: Sentence) -> float:
        """Return log P(sentence) using natural logarithm.

        If any conditional probability is zero, log probability is -inf.
        """
        self._check_fitted()
        tokens = tokenize(sentence)

        if not tokens:
            return 0.0

        log_probability = 0.0

        for i, word in enumerate(tokens):
            start = max(0, i - (self.n - 1))
            context = tokens[start:i]
            p = self.probability(context, word)

            if p == 0.0:
                return float("-inf")

            log_probability += math.log(p)

        return log_probability

    def next_word_distribution(self, context: Sentence) -> Dict[str, float]:
        """Return P(next_word | context) for every vocabulary word."""
        self._check_fitted()
        distribution = {}

        for word in sorted(self.vocabulary):
            distribution[word] = self.probability(context, word)

        return distribution

    def predict_next(self, context: Sentence, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k next-word predictions."""
        if top_k < 1:
            return []

        distribution = self.next_word_distribution(context)
        ranked = sorted(
            distribution.items(),
            key=lambda item: (-item[1], item[0])
        )
        return ranked[:top_k]

    def perplexity(self, data: Union[Sentence, Corpus]) -> float:
        """Compute perplexity.

        Accepted input:
        - one sentence as a string: "the cat eats fish"
        - one tokenized sentence: ["the", "cat", "eats", "fish"]
        - a corpus: ["sentence one", "sentence two"]
        - a tokenized corpus: [[...], [...]]

        Perplexity = exp(- total_log_probability / number_of_tokens)
        If any required probability is zero, perplexity is +inf.
        """
        self._check_fitted()

        # One string = one sentence.
        if isinstance(data, str):
            sentences = [tokenize(data)]
        else:
            data_list = list(data)

            if not data_list:
                raise ValueError("Cannot compute perplexity of empty data")

            # A flat sequence of strings is ambiguous. If all elements contain
            # no whitespace, treat it as one tokenized sentence. Otherwise,
            # treat it as a corpus of sentence strings.
            if all(isinstance(x, str) for x in data_list):
                if all(len(x.split()) == 1 for x in data_list):
                    sentences = [list(data_list)]
                else:
                    sentences = [tokenize(x) for x in data_list]
            else:
                sentences = [tokenize(x) for x in data_list]

        total_log_probability = 0.0
        total_tokens = 0

        for sentence in sentences:
            if not sentence:
                continue

            log_p = self.sentence_log_probability(sentence)
            if log_p == float("-inf"):
                return float("inf")

            total_log_probability += log_p
            total_tokens += len(sentence)

        if total_tokens == 0:
            raise ValueError("Cannot compute perplexity with zero tokens")

        return math.exp(-total_log_probability / total_tokens)

    def continuation_log_probability(
        self,
        context: Sentence,
        continuation: Sentence
    ) -> float:
        """Score only a continuation conditioned on an existing context.

        Useful for the sentence-ranking application P(candidate | context).
        """
        self._check_fitted()

        context_tokens = tokenize(context)
        continuation_tokens = tokenize(continuation)
        history = list(context_tokens)
        log_probability = 0.0

        for word in continuation_tokens:
            if self.n == 1:
                local_context = []
            else:
                local_context = history[-(self.n - 1):]

            p = self.probability(local_context, word)
            if p == 0.0:
                return float("-inf")

            log_probability += math.log(p)
            history.append(word)

        return log_probability

    def rank_candidates(
        self,
        context: Sentence,
        candidates: Sequence[Sentence]
    ) -> List[Tuple[Sentence, float]]:
        """Rank candidate continuations by conditional log probability."""
        scored = []

        for candidate in candidates:
            score = self.continuation_log_probability(context, candidate)
            scored.append((candidate, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)
