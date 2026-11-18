"""Mục 3, 5, 13, 18–20 — CoNLL-2003 và baseline Logistic Regression."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

SEED = 42
SPLIT_FILES = {"train": "eng.train", "validation": "eng.testa", "test": "eng.testb"}
SPLIT_SIZES = {"train": 14041, "validation": 3250, "test": 3453}
CORPUS = "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/"
TYPES = ("PER", "ORG", "LOC", "MISC")
LABELS = ("O", *(tag + "-" + entity for entity in TYPES for tag in ("B", "I")))
LABEL2ID = {label: index for index, label in enumerate(LABELS)}


def bio_spans(tags):
    """Return (start, end_exclusive, type); orphan I-X starts a fresh X span."""
    spans = []
    start, entity = 0, None
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, incoming = tag.partition("-")
        if prefix == "I" and incoming == entity:
            continue
        if entity is not None:
            spans.append((start, i, entity))
        start, entity = i, incoming if prefix in {"B", "I"} else None
    return spans


def normalize_bio(tags):
    normalized = ["O"] * len(tags)
    for start, end, entity in bio_spans(tags):
        normalized[start] = f"B-{entity}"
        normalized[start + 1:end] = [f"I-{entity}"] * (end - start - 1)
    return normalized


def read_conll(path):
    rows, words, tags = [], [], []
    for line in [*Path(path).read_text(encoding="utf-8-sig").splitlines(), ""]:
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            if words:
                unknown = set(tags).difference(LABELS)
                if unknown:
                    raise ValueError(f"{path}: nhãn chưa biết {unknown}")
                rows.append({"words": words, "tags": normalize_bio(tags)})
                words, tags = [], []
            continue
        word, _, _, tag = line.split()
        words.append(word)
        tags.append(tag)
    return rows


def load_corpus(data_dir=None):
    """Load the official English train/dev/test split from a CoNLL-format mirror."""
    data_dir = Path(data_dir or os.getenv("LAB09_DATA_DIR", ".cache/conll2003"))
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus = {}
    for split, name in SPLIT_FILES.items():
        path = data_dir / name
        if not path.is_file():
            request = Request(CORPUS + name, headers={"User-Agent": "LAB09-CoNLL/1.0"})
            with urlopen(request, timeout=90) as response:
                content = response.read()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(content)
            temporary.replace(path)
        rows = read_conll(path)
        if len(rows) != SPLIT_SIZES[split]:
            raise ValueError(f"{split}: tìm thấy {len(rows)} câu, cần {SPLIT_SIZES[split]} trong {path}.")
        corpus[split] = rows
    return corpus


def word_shape(word):
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
                   for c in word)


def token_features(words, index):
    """Local surface features, without POS/chunk tags or corpus gold as inputs."""
    word = words[index]
    lower = word.casefold()
    features = {
        "word": lower, "shape": word_shape(word), "title": word.istitle(),
        "upper": word.isupper(), "digit": any(char.isdigit() for char in word),
        "punct": all(not char.isalnum() for char in word),
        "length": str(min(len(word), 12)),
    }
    for width in (1, 2, 3, 4):
        features[f"prefix{width}"] = lower[:width]
        features[f"suffix{width}"] = lower[-width:]
    for offset, name in ((-1, "prev"), (1, "next")):
        neighbor_index = index + offset
        if 0 <= neighbor_index < len(words):
            neighbor = words[neighbor_index]
            features[f"{name}_word"] = neighbor.casefold()
            features[f"{name}_shape"] = word_shape(neighbor)
        else:
            features[f"{name}_boundary"] = True
    return features


def featurize(rows):
    return (token_features(row["words"], i) for row in rows for i in range(len(row["words"])))


def train_baseline(train):
    vectorizer = DictVectorizer(dtype=np.float64)
    inputs = vectorizer.fit_transform(featurize(train))
    targets = [tag for row in train for tag in row["tags"]]
    model = LogisticRegression(C=2.0, solver="lbfgs", max_iter=220, tol=0.002, random_state=SEED)
    with threadpool_limits(limits=3):
        model.fit(inputs, targets)
    return vectorizer, model


def predict_baseline(fitted, examples):
    vectorizer, model = fitted
    labels = model.predict(vectorizer.transform(featurize(examples))).tolist()
    cursor, predictions = 0, []
    for row in examples:
        end = cursor + len(row["words"])
        predictions.append(normalize_bio(labels[cursor:end]))
        cursor = end
    return predictions


def entity_set(sequences):
    return {(i, start, end, entity) for i, tags in enumerate(sequences)
            for start, end, entity in bio_spans(tags)}


def word_label_set(sequences):
    return {(i, position, tag) for i, tags in enumerate(sequences)
            for position, tag in enumerate(tags) if tag != "O"}


def score_sets(gold, predicted):
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "support": len(gold),
            "precision": precision, "recall": recall, "f1": f1}


def score_sequences(gold, predicted):
    if [len(tags) for tags in gold] != [len(tags) for tags in predicted]:
        raise ValueError("Số câu hoặc số từ giữa gold và prediction không khớp.")
    return {
        "entity_exact_micro": score_sets(entity_set(gold), entity_set(predicted)),
        "word_bio_micro_no_o": score_sets(word_label_set(gold), word_label_set(predicted)),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cần ít nhất một dòng để xác lập cột CSV.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Mục 13 — chạy baseline NER trên CoNLL-2003.")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    corpus = load_corpus(args.data_dir)
    predictions = predict_baseline(train_baseline(corpus["train"]), corpus["test"])
    gold = [row["tags"] for row in corpus["test"]]
    for level, scores in score_sequences(gold, predictions).items():
        print(level, scores)


if __name__ == "__main__":
    main()
