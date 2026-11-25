"""§27 – Extractive QA on an English plain-text document supplied by the caller."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from qa_model import Example, ExtractiveBert, Settings, load_tokenizer, make_features, predict_examples

DEMO_DOCUMENT = (
    "The Northbridge Library is located in Boston. "
    "It opens at 9 am on weekdays. "
    "Library cards are valid for two years. "
    "The main reading room closes at 6 pm."
)
DEMO_QUESTIONS = (
    "Where is the Northbridge Library located?",
    "When does the Northbridge Library open on weekdays?",
    "How long are library cards valid?",
    "Who designed the Northbridge Library?",
)


def load_reader(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}. Run §17 in qa_finetuning.ipynb.")
    run_file = checkpoint / "run.json"
    run = json.loads(run_file.read_text(encoding="utf-8")) if run_file.is_file() else {}
    settings = Settings(**run.get("settings", {}))
    tokenizer = load_tokenizer(checkpoint)
    model = ExtractiveBert.from_pretrained(checkpoint).to(device).eval()
    return model, tokenizer, settings


def answer_questions(model, tokenizer, settings, context, questions):
    if not context.strip():
        raise ValueError("Document must contain text.")
    if any(not question.strip() for question in questions):
        raise ValueError("Each question must contain text.")
    examples = [Example(str(i), question, context, ()) for i, question in enumerate(questions)]
    features = make_features(examples, tokenizer, settings.max_length, settings.stride,
                             settings.max_question_tokens)
    predictions, _ = predict_examples(model, tokenizer, examples, features, settings,
                                      compute_loss=False)
    return [predictions[example.id] for example in examples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lab_dir = Path(__file__).resolve().parent
    checkpoint = lab_dir / ".lab10" / "runs" / "384"
    bundled = lab_dir.parent / "checkpoint_384"
    if not checkpoint.is_dir() and bundled.is_dir():
        checkpoint = bundled
    parser.add_argument("--model", type=Path, default=checkpoint)
    parser.add_argument("--document", type=Path, help="UTF-8 text document")
    parser.add_argument("--question", action="append", help="Repeat for multiple questions")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))
    try:
        if args.demo:
            if args.document or args.question:
                parser.error("--demo uses the built-in document and questions")
            context, questions = DEMO_DOCUMENT, DEMO_QUESTIONS
        else:
            if args.document is None or args.question is None:
                parser.error("provide --document and --question, or use --demo")
            context = args.document.read_text(encoding="utf-8")
            questions = args.question
        model, tokenizer, settings = load_reader(args.model, device)
        settings = replace(settings, batch_size=min(settings.batch_size, 16))
        spans = answer_questions(model, tokenizer, settings, context, questions)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for question, span in zip(questions, spans):
        if span.start is None:
            print(json.dumps({"question": question, "answer": "", "start": None, "end": None}))
            continue
        print(json.dumps(
            {"question": question, "answer": span.text, "start": span.start,
             "end_exclusive": span.end, "window": span.window + 1,
             "logit_margin": round(span.score, 4)},
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
