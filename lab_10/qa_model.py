"""§4–5, §8, §13–21, §26 – Extractive QA, from offsets to span metrics."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import random
import re
import string
from time import perf_counter
from urllib.request import urlretrieve

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset"
MODEL_KEYS = ("input_ids", "attention_mask", "token_type_ids")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    context: str
    answers: tuple  # (text, start) pairs
    title: str = ""


@dataclass(frozen=True)
class Feature:
    example_id: str
    inputs: dict
    offsets: tuple  # Context-only half-open character offsets; None elsewhere.
    cls_index: int
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    text: str
    start: int | None
    end: int | None
    score: float
    window: int | None = None


@dataclass(frozen=True)
class Settings:
    max_length: int = 384
    stride: int = 64
    batch_size: int = 32
    learning_rate: float = 2e-4
    epochs: int = 3
    seed: int = 42
    max_answer_tokens: int = 30
    max_question_tokens: int = 48


def load_tokenizer(source=MODEL_ID):
    args = {"revision": MODEL_REVISION} if str(source) == MODEL_ID else {}
    return AutoTokenizer.from_pretrained(source, use_fast=True, **args)


def load_squad(directory, train_count=4200, val_count=600, test_count=1200, seed=42):
    """§3, §30 – Keep official train separate; split official dev by article."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def read_split(filename):
        path = directory / filename
        if not path.is_file():
            temporary = path.with_suffix(".download")
            urlretrieve(f"{SQUAD_URL}/{filename}", temporary)
            temporary.replace(path)
        articles = json.loads(path.read_text(encoding="utf-8"))["data"]
        return {
            article["title"]: [
                Example(
                    row["id"], row["question"], paragraph["context"],
                    tuple((answer["text"], answer["answer_start"])
                          for answer in row["answers"]),
                    article["title"],
                )
                for paragraph in article["paragraphs"] for row in paragraph["qas"]
            ]
            for article in articles
        }

    def sample(pool, count):
        return pool if count is None else random.Random(seed).sample(pool, min(count, len(pool)))

    official_train = read_split("train-v1.1.json")
    official_dev = read_split("dev-v1.1.json")
    titles = sorted(official_dev)
    random.Random(seed).shuffle(titles)
    cut = len(titles) // 2
    train = [item for rows in official_train.values() for item in rows]
    validation = [item for title in titles[:cut] for item in official_dev[title]]
    test = [item for title in titles[cut:] for item in official_dev[title]]
    splits = {
        "train": sample(train, train_count),
        "validation": sample(validation, val_count),
        "test": sample(test, test_count),
    }
    ids = {key: {item.id for item in examples} for key, examples in splits.items()}
    assert ids["train"].isdisjoint(ids["validation"] | ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert set(official_train).isdisjoint(official_dev)
    return splits


# §10–16 – Preprocess pairs and align inclusive token boundaries.
def tokenize_pairs(tokenizer, questions, contexts, max_length, stride, max_question_tokens=48):
    questions = [question.lstrip() for question in questions]
    if max_question_tokens is not None:
        question_offsets = tokenizer(questions, add_special_tokens=False,
                                     return_offsets_mapping=True, verbose=False)["offset_mapping"]
        questions = [
            question[:offsets[max_question_tokens - 1][1]]
            if len(offsets) > max_question_tokens else question
            for question, offsets in zip(questions, question_offsets)
        ]
    return tokenizer(
        questions, contexts, truncation="only_second", max_length=max_length,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
    )


def align_answer(offsets, answer_start, answer_end):
    """Return inclusive token indices, or None if the answer is outside the window."""
    start_token = end_token = None
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        left, right = offset
        if left <= answer_start < right:
            start_token = index
        if left < answer_end <= right:
            end_token = index
    if start_token is None or end_token is None or start_token > end_token:
        return None
    return start_token, end_token


def make_features(examples, tokenizer, max_length, stride, max_question_tokens=48):
    features = []
    for start in range(0, len(examples), 256):
        examples_batch = examples[start:start + 256]
        encoded = tokenize_pairs(tokenizer, [row.question for row in examples_batch],
                                 [row.context for row in examples_batch], max_length,
                                 stride, max_question_tokens)
        for index, sample_index in enumerate(encoded["overflow_to_sample_mapping"]):
            example = examples_batch[sample_index]
            inputs = {key: encoded[key][index] for key in MODEL_KEYS if key in encoded}
            offsets = tuple(
                tuple(offset) if part == 1 and offset[0] < offset[1] else None
                for part, offset in zip(encoded.sequence_ids(index), encoded["offset_mapping"][index])
            )
            cls = inputs["input_ids"].index(tokenizer.cls_token_id)
            aligned = (cls, cls)
            for text, char_start in example.answers:
                char_end = char_start + len(text)
                if not text or example.context[char_start:char_end] != text:
                    continue
                candidate = align_answer(offsets, char_start, char_end)
                if candidate is not None:
                    aligned = candidate
                    break
            features.append(Feature(example.id, inputs, offsets, cls, *aligned))
    return features


# §5, §8, §17 – Pretrained encoder + one two-channel projection.
class ExtractiveBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_head = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, span_mask=None,
                start_positions=None, end_positions=None):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state
        start, end = self.qa_head(hidden).unbind(dim=-1)
        if span_mask is not None:
            start = start.masked_fill(~span_mask, torch.finfo(start.dtype).min)
            end = end.masked_fill(~span_mask, torch.finfo(end.dtype).min)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (cross_entropy(start, start_positions) + cross_entropy(end, end_positions)) / 2
        return QuestionAnsweringModelOutput(loss=loss, start_logits=start, end_logits=end)


def collate(features, tokenizer):
    inputs = tokenizer.pad([feature.inputs for feature in features], return_tensors="pt")
    span_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    for row, feature in enumerate(features):
        span_mask[row, :len(feature.offsets)] = torch.tensor(
            [offset is not None or i == feature.cls_index
             for i, offset in enumerate(feature.offsets)], dtype=torch.bool,
        )
    inputs["span_mask"] = span_mask
    inputs["start_positions"] = torch.tensor([feature.start for feature in features])
    inputs["end_positions"] = torch.tensor([feature.end for feature in features])
    return inputs


def best_window_span(feature, start_logits, end_logits, max_answer_tokens):
    """Exact search over start <= end and spans of at most max_answer_tokens."""
    length = len(feature.offsets)
    if not any(offset is not None for offset in feature.offsets):
        return None
    starts = torch.arange(length)[:, None]
    ends = starts + torch.arange(max_answer_tokens)[None, :]
    in_bounds = ends < length
    ends = ends.clamp(max=length - 1)
    context_mask = torch.tensor([offset is not None for offset in feature.offsets])
    valid = in_bounds & context_mask[starts] & context_mask[ends]
    scores = start_logits[:length, None] + end_logits[ends]
    best = scores.masked_fill(~valid, -torch.inf).flatten().argmax().item()
    one, two = divmod(best, max_answer_tokens)
    two += one
    null_score = start_logits[feature.cls_index] + end_logits[feature.cls_index]
    score = (start_logits[one] + end_logits[two] - null_score).item()
    return feature.offsets[one][0], feature.offsets[two][1], score


# §18–20 – SQuAD v1 normalization and multi-reference scoring.
ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalized_tokens(text):
    return ARTICLE_RE.sub(" ", text.lower().translate(PUNCTUATION)).split()


def exact_match(prediction, gold):
    return float(normalized_tokens(prediction) == normalized_tokens(gold))


def token_f1(prediction, gold):
    predicted, reference = normalized_tokens(prediction), normalized_tokens(gold)
    if not predicted or not reference:
        return float(predicted == reference)
    shared = sum((Counter(predicted) & Counter(reference)).values())
    return 2 * shared / (len(predicted) + len(reference))


def score_predictions(examples, predictions):
    em = f1 = 0.0
    for example in examples:
        golds = [gold for gold, _ in example.answers] or [""]
        predicted = predictions[example.id].text
        em += max(exact_match(predicted, gold) for gold in golds)
        f1 += max(token_f1(predicted, gold) for gold in golds)
    return {"em": 100 * em / len(examples), "f1": 100 * f1 / len(examples)}


def predict_examples(model, tokenizer, examples, features, settings, device=None,
                     null_threshold=None, compute_loss=True):
    """§17–20, §26 – Pool by example using the score margin against each window's CLS."""
    device = device or next(model.parameters()).device
    model.eval()
    contexts = {example.id: example.context for example in examples}
    best = {}
    windows = defaultdict(int)
    loss_sum = 0.0
    with torch.inference_mode():
        for offset in range(0, len(features), settings.batch_size):
            batch = features[offset:offset + settings.batch_size]
            tensors = collate(batch, tokenizer).to(device)
            if not compute_loss:
                tensors.pop("start_positions")
                tensors.pop("end_positions")
            output = model(**tensors)
            if compute_loss:
                loss_sum += output.loss.item() * len(batch)
            starts, ends = output.start_logits.cpu(), output.end_logits.cpu()
            for index, feature in enumerate(batch):
                window = windows[feature.example_id]
                windows[feature.example_id] += 1
                candidate = best_window_span(feature, starts[index], ends[index],
                                             settings.max_answer_tokens)
                if candidate is None:
                    continue
                left, right, score = candidate
                previous = best.get(feature.example_id)
                if previous is None or score > previous.score:
                    best[feature.example_id] = Span(contexts[feature.example_id][left:right],
                                                    left, right, score, window)
    for example in examples:
        prediction = best.get(example.id, Span("", None, None, -math.inf))
        if null_threshold is not None and prediction.score < null_threshold:
            prediction = Span("", None, None, prediction.score, prediction.window)
        best[example.id] = prediction
    return best, (loss_sum / len(features) if compute_loss and features else math.nan)


# §21 – Lexical overlap sentence retrieval + a deterministic contiguous candidate.
WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b")
SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|\Z)", re.S)
STOP = set("the a an in of and it by on to at are is was were what who when "
           "where which how for from that did does do has have this had with".split())
CONNECTORS = {"of", "the", "and", "de", "van", "da", "von"}
DATE_RE = re.compile(
    r"\b(?:(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December)\s+\d{1,2}(?:,\s*\d{4})?|\d{4})\b", re.I,
)
NUMBER_RE = re.compile(
    r"\b\d[\d,.]*(?:\s+(?:million|billion|thousand|percent))?%?(?!\w)", re.I,
)
NAME_RE = re.compile(
    r"\b[A-Z][\w'’-]*(?:\s+(?:(?:of|the|and|de|van|da|von)\s+)?[A-Z][\w'’-]*)*"
)


def content_words(text):
    return {word.lower() for word in WORD_RE.findall(text)} - STOP


def baseline_predict(example):
    keywords = content_words(example.question)
    sentences = list(SENTENCE_RE.finditer(example.context))
    if not sentences:
        return Span("", None, None, 0.0)
    sentence = max(sentences, key=lambda candidate: len(keywords & content_words(candidate[0])))
    question = example.question.lstrip().lower()
    pattern = NAME_RE
    if question.startswith("when") or re.search(r"\b(what|which) year\b", question):
        pattern = DATE_RE
    elif question.startswith(("how many", "how much", "how long", "how old")):
        pattern = NUMBER_RE
    for candidate in pattern.finditer(sentence[0]):
        words = content_words(candidate[0]) - CONNECTORS
        if words - keywords or pattern is not NAME_RE:
            start = sentence.start() + candidate.start()
            return Span(candidate[0], start, sentence.start() + candidate.end(), 0.0)
    for candidate in WORD_RE.finditer(sentence[0]):
        if candidate[0].lower() not in keywords | STOP:
            return Span(candidate[0], sentence.start() + candidate.start(),
                        sentence.start() + candidate.end(), 0.0)
    return Span(sentence[0], sentence.start(), sentence.end(), 0.0)


# §14, §17, §30 – Shared, self-contained training entry point for the notebook.
def fit(model_source, tokenizer, train_features, validation_examples, validation_features,
        settings, device, on_epoch=print):
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    source_args = {"revision": MODEL_REVISION} if str(model_source) == MODEL_ID else {}
    model = ExtractiveBert.from_pretrained(model_source, **source_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    loader = DataLoader(
        train_features, batch_size=settings.batch_size, shuffle=True,
        collate_fn=lambda features: collate(features, tokenizer),
        generator=torch.Generator().manual_seed(settings.seed),
    )
    history = []
    best_state, best_f1 = None, -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        start = perf_counter()
        train_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item() * len(batch["input_ids"])
        elapsed = perf_counter() - start
        predictions, validation_loss = predict_examples(
            model, tokenizer, validation_examples, validation_features, settings, device,
        )
        metrics = score_predictions(validation_examples, predictions)
        record = {
            "epoch": epoch, "train_loss": train_loss / len(train_features),
            "validation_loss": validation_loss, **metrics, "train_seconds": elapsed,
        }
        history.append(record)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {name: item.detach().cpu().clone() for name, item in model.state_dict().items()}
        on_epoch(record)
    model.load_state_dict(best_state)
    return model, history


def save_run(directory, model, tokenizer, meta, settings, history):
    directory = Path(directory)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    (directory / "run.json").write_text(
        json.dumps({"metadata": meta, "settings": asdict(settings), "history": history}, indent=2),
        encoding="utf-8",
    )
