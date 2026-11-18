"""Mục 25–27 — News IE bằng checkpoint NER và 30 câu gold gán theo span ký tự."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baseline_ner import bio_spans, score_sets

OUTPUT_TYPES = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}
RAW_WORD = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
MAX_LENGTH = 256
STRIDE = 64
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoint"

# Mục 26–27. Câu news mô phỏng do người viết tạo cho lab, ngoài CoNLL.
# Gold được khai báo theo cụm ký tự nguyên gốc, độc lập với dự đoán.
NEWS_GOLD = [
    ("Barack Obama visited Paris after meeting officials from UNESCO.", [
        ("Barack Obama", "PERSON"), ("Paris", "LOCATION"), ("UNESCO", "ORG")]),
    ("Microsoft opened a research center in Dublin.", [
        ("Microsoft", "ORG"), ("Dublin", "LOCATION")]),
    ("Anita Rao joined the World Bank in Washington.", [
        ("Anita Rao", "PERSON"), ("World Bank", "ORG"), ("Washington", "LOCATION")]),
    ("Reuters reporter Maria Chen arrived in Brussels on Monday.", [
        ("Reuters", "ORG"), ("Maria Chen", "PERSON"), ("Brussels", "LOCATION")]),
    ("The United Nations Security Council met in New York.", [
        ("United Nations Security Council", "ORG"), ("New York", "LOCATION")]),
    ("Barcelona defeated Valencia before returning to Barcelona for training.", [
        ("Barcelona", "ORG", 0), ("Valencia", "ORG"), ("Barcelona", "LOCATION", 1)]),
    ("Cambridge University Press announced a partnership in Cambridge.", [
        ("Cambridge University Press", "ORG"), ("Cambridge", "LOCATION", 1)]),
    ("Priya Desai addressed the audience in Nairobi.", [
        ("Priya Desai", "PERSON"), ("Nairobi", "LOCATION")]),
    ("Toyota and Panasonic announced a joint project in Osaka.", [
        ("Toyota", "ORG"), ("Panasonic", "ORG"), ("Osaka", "LOCATION")]),
    ("Daniel Kim met Sarah Patel in San Francisco.", [
        ("Daniel Kim", "PERSON"), ("Sarah Patel", "PERSON"), ("San Francisco", "LOCATION")]),
    ("The committee will publish its findings next week.", []),
    ("A power outage delayed several trains on Tuesday.", []),
    ("Basel hosted delegates from the European Central Bank.", [
        ("Basel", "LOCATION"), ("European Central Bank", "ORG")]),
    ("The BBC interviewed Omar Haddad in Amman.", [
        ("BBC", "ORG"), ("Omar Haddad", "PERSON"), ("Amman", "LOCATION")]),
    ("Liverpool beat Arsenal in London.", [
        ("Liverpool", "ORG"), ("Arsenal", "ORG"), ("London", "LOCATION")]),
    ("Medecins Sans Frontieres sent a team to Port-au-Prince.", [
        ("Medecins Sans Frontieres", "ORG"), ("Port-au-Prince", "LOCATION")]),
    ("Elena García traveled from Madrid to Buenos Aires.", [
        ("Elena García", "PERSON"), ("Madrid", "LOCATION"), ("Buenos Aires", "LOCATION")]),
    ("A spokesperson said the talks would resume before noon.", []),
    ("The International Monetary Fund released a report on Ghana.", [
        ("International Monetary Fund", "ORG"), ("Ghana", "LOCATION")]),
    ("Abdul Rahman announced the plan during a visit to Kuala Lumpur.", [
        ("Abdul Rahman", "PERSON"), ("Kuala Lumpur", "LOCATION")]),
    ("Eva O'Connor spoke with officials from UNICEF in Geneva.", [
        ("Eva O'Connor", "PERSON"), ("UNICEF", "ORG"), ("Geneva", "LOCATION")]),
    ("Samsung Electronics expanded its operations in South Korea.", [
        ("Samsung Electronics", "ORG"), ("South Korea", "LOCATION")]),
    ("Manchester United hired Alex Morgan as an adviser.", [
        ("Manchester United", "ORG"), ("Alex Morgan", "PERSON")]),
    ("The summit brought researchers from Cape Town and Singapore together.", [
        ("Cape Town", "LOCATION"), ("Singapore", "LOCATION")]),
    ("The court postponed the hearing until Friday.", []),
    ("Linh Nguyen joined the interview from Ho Chi Minh City.", [
        ("Linh Nguyen", "PERSON"), ("Ho Chi Minh City", "LOCATION")]),
    ("Pierre Dubois resigned from Air France in a statement issued in Lyon.", [
        ("Pierre Dubois", "PERSON"), ("Air France", "ORG"), ("Lyon", "LOCATION")]),
    ("NASA briefed reporters in Houston after the launch.", [
        ("NASA", "ORG"), ("Houston", "LOCATION")]),
    ("Aisha Bello moved from Lagos to Abuja last year.", [
        ("Aisha Bello", "PERSON"), ("Lagos", "LOCATION"), ("Abuja", "LOCATION")]),
    ("Toronto signed a new goalkeeper before the match in Toronto.", [
        ("Toronto", "ORG", 0), ("Toronto", "LOCATION", 1)]),
]


def gold_spans(text, annotations):
    spans = set()
    for surface, label, *indices in annotations:
        starts = [match.start() for match in re.finditer(
            rf"(?<!\w){re.escape(surface)}(?!\w)", text
        )]
        if indices and indices[0] < len(starts):
            start = starts[indices[0]]
        elif not indices and len(starts) == 1:
            start = starts[0]
        else:
            raise ValueError(f"Gold không giải được vị trí {surface!r} trong {text!r}")
        spans.add((start, start + len(surface), label))
    return spans


class NewsExtractor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(self.device).eval()

    def extract(self, text):
        """Mục 25 — NER trên raw text, khôi phục span ký tự nửa mở [start, end)."""
        words = list(RAW_WORD.finditer(text))
        if not words:
            return []
        encoded = self.tokenizer(
            [word.group() for word in words],
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            truncation=True,
            stride=STRIDE,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        # Chọn cửa sổ có nhiều context hai phía nhất nếu một từ có mặt nhiều lần.
        best = {}
        for window, ids in enumerate(encoded["input_ids"]):
            tensor_inputs = {
                key: torch.tensor([encoded[key][window]], device=self.device)
                for key in self.tokenizer.model_input_names
            }
            with torch.inference_mode():
                predicted = self.model(**tensor_inputs).logits[0].argmax(dim=-1).tolist()
            previous = None
            for i, (word_id, (offset, _)) in enumerate(
                zip(encoded.word_ids(window), encoded["offset_mapping"][window])
            ):
                if word_id is not None and word_id != previous and offset == 0:
                    context = min(i - 1, len(ids) - 2 - i)
                    if word_id not in best or context > best[word_id][0]:
                        best[word_id] = (context, self.model.config.id2label[predicted[i]])
                previous = word_id

        tags = [best.get(i, (0, "O"))[1] for i in range(len(words))]
        entities = []
        for first, end_word, entity in bio_spans(tags):
            if entity in OUTPUT_TYPES:
                start, end = words[first].start(), words[end_word - 1].end()
                entities.append({
                    "text": text[start:end], "type": OUTPUT_TYPES[entity],
                    "start": start, "end": end,
                })
        return entities

    def structured(self, text):
        by_type = defaultdict(list)
        for entity in self.extract(text):
            by_type[entity["type"]].append(entity["text"])
        return dict(by_type)


def evaluate(extractor, data=NEWS_GOLD):
    """Mục 27 — exact (sentence ID, char start, char end, type)."""
    gold, predicted, details = set(), set(), []
    for sentence, (text, annotations) in enumerate(data, start=1):
        expected = gold_spans(text, annotations)
        output = {(r["start"], r["end"], r["type"]) for r in extractor.extract(text)}
        gold.update((sentence, *span) for span in expected)
        predicted.update((sentence, *span) for span in output)
        details.append({
            "sentence_id": sentence,
            "text": text,
            "gold": [f"{text[a:b]} [{kind}]" for a, b, kind in sorted(expected)],
            "prediction": [f"{text[a:b]} [{kind}]" for a, b, kind in sorted(output)],
            "exact": expected == output,
        })
    scores = [{"entity_type": "MICRO", **score_sets(gold, predicted)}]
    for entity in OUTPUT_TYPES.values():
        scores.append({
            "entity_type": entity,
            **score_sets({span for span in gold if span[-1] == entity},
                         {span for span in predicted if span[-1] == entity}),
        })
    return scores, details


def main():
    parser = argparse.ArgumentParser(description="LAB 09 — Mục 25–27: News IE.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--text", help="Câu hoặc đoạn văn tiếng Anh.")
    task.add_argument("--evaluate", action="store_true", help="Chấm 30 câu có gold.")
    parser.add_argument("--spans", action="store_true", help="Xuất offset ký tự khi dùng --text.")
    parser.add_argument("--details", action="store_true", help="Kèm từng câu khi dùng --evaluate.")
    args = parser.parse_args()
    extractor = NewsExtractor(args.checkpoint, args.device)
    if args.evaluate:
        scores, details = evaluate(extractor)
        output = {"num_sentences": len(details), "scores": scores}
        if args.details:
            output["examples"] = details
    else:
        output = extractor.extract(args.text) if args.spans else extractor.structured(args.text)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
