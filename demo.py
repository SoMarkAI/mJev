"""Run native Omni / Qwen3-VL media candidate scoring; use --check-only for tokenizer preflight."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local official model directory')
    parser.add_argument('--input', default='examples/single.json')
    parser.add_argument('--mode', choices=['isolated', 'causal'], default='causal')
    parser.add_argument('--output', default='outputs/demo.json')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--tensor-parallel-size',type=int,default=None,help='Default: VL=1, Omni=4')
    args = parser.parse_args()
    from mjev.inputs import normalize_task
    source = Path(args.input).resolve()
    task = normalize_task(json.loads(source.read_text()), source.parent, default_context='')
    # Validate task structure before importing runtime hooks or loading a model.
    root = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = root + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['MJEV_ENABLE'] = '1'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    from mjev.patch import install
    install()
    from mjev.protocol import Protocol
    image = task['media']
    modality, video_options = task['modality'], task['video_options']
    questions, context = task['questions'], task['context']
    protocol = Protocol(args.model)
    checked = []
    for q in questions:
        _, _, labels, ids = protocol.build(image, context, q['question'], q['candidates'], args.mode,modality=modality,video_options=video_options)
        checked.append({'question': q['question'], 'labels': labels, 'token_ids': ids})
    if args.check_only:
        result = {'check_only': True, 'inference_performed': False, 'questions': checked}
    else:
        from mjev.engine import MJevEngine
        async def run():
            engine = MJevEngine(args.model,enable_av=modality!='image',tensor_parallel_size=args.tensor_parallel_size)
            try:
                return await asyncio.gather(*(engine.score_media(image,q['question'],q['candidates'],
                    modality=modality,context=context,mode=args.mode,video_options=video_options) for q in questions))
            finally:
                engine.close()
        result = asyncio.run(run())
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Result: {target}', file=sys.stderr)

if __name__ == '__main__':
    main()
