"""Mục 16 — Alignment, mã hóa và dự đoán NER theo từ."""
import random

import torch
from transformers import AutoModelForTokenClassification

from baseline_ner import LABEL2ID, LABELS, SEED, normalize_bio

IGNORE = -100


def align_first_pieces(word_ids, word_tags):
    """Mục 9–11 — Strategy A, cùng cách gán cho cả ba split."""
    targets, kept = [], []
    previous = None
    for index, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous:
            targets.append(IGNORE)
        else:
            targets.append(LABEL2ID[word_tags[word_id]])
            kept.append(index)
        previous = word_id
    return targets, kept


def encode_rows(rows, tokenizer, max_length):
    batch = tokenizer(
        [row["words"] for row in rows],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = []
    for i, row in enumerate(rows):
        word_ids = batch.word_ids(i)
        targets, kept = align_first_pieces(word_ids, row["tags"])
        if [word_ids[index] for index in kept] != list(range(len(row["words"]))):
            raise ValueError("Truncation đã làm mất từ. Tăng MAX_LENGTH trước khi so sánh cùng test set.")
        encoded.append({**{key: batch[key][i] for key in tokenizer.model_input_names},
                        "labels": targets})
    return encoded


def make_model(model_source, device):
    return AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABELS),
        id2label={i: tag for i, tag in enumerate(LABELS)},
        label2id=LABEL2ID,
        use_safetensors=True,
    ).to(device)


def predict_words(model, items, collate, device, batch_size):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size])
            valid = batch.pop("labels").ne(IGNORE)
            logits = model(**batch.to(device)).logits
            for prediction, mask in zip(logits.argmax(dim=-1).cpu(), valid):
                result.append(normalize_bio([LABELS[i] for i in prediction[mask].tolist()]))
    return result


def epoch_batches(items, epoch, batch_size):
    rng = random.Random(SEED + epoch)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    blocks = []
    for start in range(0, len(indices), batch_size * 16):
        group = indices[start:start + batch_size * 16]
        group.sort(key=lambda i: len(items[i]["input_ids"]))
        blocks.extend(group[i:i + batch_size] for i in range(0, len(group), batch_size))
    rng.shuffle(blocks)
    return blocks
