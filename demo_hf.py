"""HF demo: native model forward; no vLLM and no generate."""
import argparse
import json
from pathlib import Path
from mjev.inputs import normalize_task


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['causal', 'isolated'], default='causal')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--projection', choices=['selected', 'full'], default='selected')
    parser.add_argument('--max-input-tokens', type=int, default=4000)
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--numerics', choices=['native', 'stable'], default='stable')
    parser.add_argument('--question-batch-size', type=int, default=1)
    return parser


def preflight(args, task):
    from mjev.hf import HFMJevEngine
    from mjev.prompt import PromptBuilder
    engine = HFMJevEngine.__new__(HFMJevEngine)
    engine.builder = PromptBuilder(args.model)
    engine.max_input_tokens = args.max_input_tokens
    results, decoded = [], None
    for question in task['questions']:
        _, spec, decoded = engine.prepare(
            task['media'], question['question'], question['candidates'],
            modality=task['modality'], context=task['context'],
            video_options=task['video_options'], decoded=decoded)
        results.append({key: spec[key] for key in ['prompt_tokens', 'spans', 'labels', 'label_ids']})
    return results


def score(args, task):
    from mjev.hf import HFMJevEngine
    engine = HFMJevEngine(args.model, device_map=args.device_map, dtype=args.dtype,
                         max_input_tokens=args.max_input_tokens, numerics=args.numerics)
    return engine.score_many(
        task['media'], task['questions'], modality=task['modality'],
        context=task['context'], video_options=task['video_options'], mode=args.mode,
        projection=args.projection, use_prefix_cache=args.prefix_cache,
        batch_size=args.question_batch_size)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.question_batch_size < 1 or args.max_input_tokens < 1:
        parser.error('batch size and input limit must be positive')
    task = normalize_task(json.loads(args.input.read_text()), args.input.parent)
    results = preflight(args, task) if args.check_only else score(args, task)
    text = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
