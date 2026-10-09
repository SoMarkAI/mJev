"""Score labelled document JSONL, or evaluate saved predictions offline."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .io import load_document, normalize_document, write_json
from .metrics import permutation_metrics, probability_metrics
from .permutations import orders
from mjev.inputs import candidate_labels


def load_benchmark(path):
    path = Path(path)
    if path.suffix == ".json":
        documents = [load_document(path)]
    else:
        documents = [
            normalize_document(json.loads(line), path.parent)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    questions, identifiers = [], set()
    for doc_index, document in enumerate(documents):
        for index, question in enumerate(document["questions"]):
            identifier = question.get("id", f"{doc_index}:{index}")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("Question IDs must be unique nonempty strings")
            identifiers.add(identifier)
            if "label" not in question:
                raise ValueError("Benchmark questions require reference labels")
            questions.append(
                {
                    **question,
                    "id": identifier,
                    "image": document["image"],
                    "context": document["context"],
                }
            )
    if not questions:
        raise ValueError("Benchmark is empty")
    return questions


def score_benchmark(
    engine,
    questions,
    protocol="none",
    random_count=4,
    seed=0,
    mode="causal",
    prefix_cache=False,
    batch_size=1,
):
    groups = defaultdict(list)
    for question in questions:
        groups[(question["image"], question["context"])].append(question)
    results = []
    for (image, context), values in groups.items():
        requests, metadata = [], []
        for question in values:
            labels = candidate_labels(len(question["candidates"]))
            target = labels.index(question["label"])
            question_seed = seed + int.from_bytes(
                hashlib.sha256(question["id"].encode()).digest()[:8]
            )
            for variant, order in enumerate(
                orders(len(labels), protocol, random_count, question_seed)
            ):
                requests.append(
                    {
                        "id": f"{question['id']}@{variant}",
                        "question": question["question"],
                        "candidates": [question["candidates"][i] for i in order],
                    }
                )
                metadata.append((question, variant, order, target))
        scored = engine.score_many(
            image,
            requests,
            context=context,
            mode=mode,
            projection="full",
            use_prefix_cache=prefix_cache,
            batch_size=batch_size,
        )
        if len(scored) != len(metadata):
            raise ValueError("Model returned the wrong number of predictions")
        for result, (question, variant, order, target) in zip(scored, metadata, strict=True):
            rows = result["candidates"]
            results.append(
                {
                    "id": question["id"],
                    "language": question.get("language", "unspecified"),
                    "variant": variant,
                    "order": list(order),
                    "target_index": order.index(target),
                    "semantic_target": target,
                    "semantic_prediction": order[result["decision"]["index"]],
                    "raw_logits": [row["raw_logit"] for row in rows],
                    "probabilities": [row["probability"] for row in rows],
                    "num_cached_tokens": result["num_cached_tokens"],
                }
            )
    return results


def validate_predictions(rows):
    for row in rows:
        count = len(row["probabilities"])
        order = row["order"]
        if sorted(order) != list(range(count)):
            raise ValueError("Saved candidate order is not a permutation")
        target = row["target_index"]
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < count:
            raise ValueError("Invalid saved target index")
        predicted = max(range(count), key=row["probabilities"].__getitem__)
        if (
            row["semantic_target"] != order[target]
            or row["semantic_prediction"] != order[predicted]
        ):
            raise ValueError("Saved semantic mapping disagrees with candidate probabilities")


def summarize(rows):
    validate_predictions(rows)
    identity = [row for row in rows if row["variant"] == 0]
    languages = sorted({row.get("language", "unspecified") for row in identity})
    return {
        "original_order": probability_metrics(identity),
        "all_variants": probability_metrics(rows),
        "permutation": permutation_metrics(rows),
        "languages": {
            lang: probability_metrics(
                [r for r in identity if r.get("language", "unspecified") == lang]
            )
            for lang in languages
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Labelled JSON or JSONL benchmark")
    source.add_argument("--predictions", type=Path, help="Saved mJev-Doc evaluation JSONL")
    parser.add_argument("--model", help="Local official model or trained checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", choices=["none", "circular", "random"], default="none")
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numerics", choices=["native", "stable"], default="native")
    parser.add_argument("--mode", choices=["causal", "isolated"], default="causal")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--question-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.question_batch_size < 1 or args.random_count < 1:
        parser.error("batch size and random count must be positive")
    if args.predictions:
        rows = [
            json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()
        ]
        metadata = {
            "source": "saved_predictions",
            "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        }
    else:
        if not args.model:
            parser.error("--model is required when scoring a benchmark")
        if (args.prefix_cache or args.question_batch_size > 1) and args.numerics != "stable":
            parser.error("Cache/batch mode requires --numerics stable")
        from .engine import DocJevEngine

        questions = load_benchmark(args.input)
        engine = DocJevEngine(args.model, numerics=args.numerics)
        rows = score_benchmark(
            engine,
            questions,
            args.protocol,
            args.random_count,
            args.seed,
            args.mode,
            args.prefix_cache,
            args.question_batch_size,
        )
        prediction_path = args.output.with_suffix(".predictions.jsonl")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        labels = Counter(question["label"] for question in questions)
        metadata = {
            "source": "live_scoring",
            "model_name": Path(args.model).name,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "protocol": args.protocol,
            "seed": args.seed,
            "mode": args.mode,
            "numerics": args.numerics,
            "prefix_cache": args.prefix_cache,
            "question_batch_size": args.question_batch_size,
            "label_histogram": dict(labels),
            "label_histogram_by_candidate_count": {
                str(count): dict(
                    Counter(q["label"] for q in questions if len(q["candidates"]) == count)
                )
                for count in sorted({len(q["candidates"]) for q in questions})
            },
        }
    report = {**metadata, **summarize(rows)}
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
