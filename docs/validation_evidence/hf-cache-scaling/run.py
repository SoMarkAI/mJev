"""Controlled HF public-video performance probe; no accuracy claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback
import torch
from mjev.hf import HFMJevEngine
from mjev.benchmark.provenance import capture
from benchmarks.integrated.validate_model import compare


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    revision = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
    questions = [
      ('How does the red ball move?', ['From left to right','From right to left','Upward','It stays still']),
      ('Which object moves upward?', ['The red ball','The blue square','Both objects']),
      ('Which movement happens first?', ['The red ball moves right','The blue square moves up','Both movements begin at the same time']),
      ('What color is the ball?', ['Red','Blue','Green','Yellow']),
      ('What shape is the blue object?', ['Square','Circle','Triangle','Star']),
      ('Does the red ball move upward?', ['Yes','No']),
      ('What color is the square?', ['Blue','Red','Yellow','Green']),
      ('Which object moves horizontally?', ['The red ball','The blue square','Both objects']),
      ('Does the blue square move after the ball starts moving?', ['Yes','No']),
      ('How many colored objects are shown?', ['One','Two','Three','Four']),
      ('Which object has a circular shape?', ['The red object','The blue object']),
      ('Does the red ball move from right to left?', ['Yes','No']),
      ('How does the blue square move?', ['Upward','Downward','From left to right','From right to left']),
      ('Which motion occurs later?', ['The square moves upward','The ball moves right','Both occur at the same time']),
      ('Is the moving circular object blue?', ['Yes','No']),
      ('Does the scene contain a moving square?', ['Yes','No']),
    ]
    qs = [dict(id=f'probe:{i}', question=q, candidates=c) for i,(q,c) in enumerate(questions)]
    profiles = {
      'long': {'nframes':24, 'min_pixels':4816896, 'max_pixels':4816896},
    }
    config = dict(model_id='Qwen/Qwen3-VL-4B-Instruct', model_revision=revision,
      code_commit=a.commit, numerics='stable', dtype='bfloat16', projection='full', mode='causal',
      seed=37, repeats=a.repeats, warmups=2, question_counts=[1,3,8,16], profiles=profiles,
      questions=qs, media_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
      scope='One project-owned synthetic motion video, varied video sampling/resolution. Performance and execution parity only; no accuracy evaluation.',
      timing='Synchronized wall time: public score_many includes decoding, processor, packing, forward and scoring; cache_batch includes new prefix construction on every call. Model loading and GC excluded.',
      memory='PyTorch CUDA peak allocated/reserved including model weights; allocator empty_cache before each measured call, outside timer. Not whole-device nvidia-smi memory.',
      harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(a.out/'config.json',config)
    torch.manual_seed(37)
    assert torch.cuda.device_count()==1
    env = capture(config,revision,a.model)
    env['code_commit']=a.commit
    env['code_commit_source']='Exact git archive deployed, harness separately hashed'
    env['model_revision_status']='Previously hash-verified local official snapshot; no download this run'
    env['gpu']=torch.cuda.get_device_name(0)
    save(a.out/'environment.json',env)
    engine=HFMJevEngine(a.model,device_map='cuda:0',dtype='bfloat16',numerics='stable',max_input_tokens=4096)
    save(a.out/'loading_info.json',engine.loading_info)
    rows=[]
    def score(profile,n,strategy):
        return engine.score_many(str(a.source),qs[:n],modality='video',context='Inspect the supplied media carefully.',
            video_options=profiles[profile],mode='causal',projection='full',
            use_prefix_cache=strategy=='cache_batch',batch_size=1 if strategy=='serial' else n)
    def measure(fn):
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated()
        t=time.perf_counter(); values=fn();torch.cuda.synchronize();seconds=time.perf_counter()-t
        return values,dict(seconds=seconds,baseline_allocated_bytes=base,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    for profile in profiles:
      for n in config['question_counts']:
        strategies=['serial','batch','cache_batch']
        failed=set(); baseline=None
        for strategy in strategies:
          try:
            for _ in range(config['warmups']):
              values,_=measure(lambda:score(profile,n,strategy))
              if baseline is None: baseline=values
              delta=compare(baseline,values)
              assert delta['max_abs_logit_delta']<=1e-4 and delta['decision_agreement']==n, delta
          except torch.cuda.OutOfMemoryError as exc:
            failed.add(strategy)
            row=dict(profile=profile,n=n,strategy=strategy,status='oom',phase='warmup',error=str(exc))
            rows.append(row);save(a.out/'raw.json',rows)
            gc.collect();torch.cuda.empty_cache()
        for rep in range(a.repeats):
          order=list(strategies);random.Random(37+rep).shuffle(order)
          for strategy in order:
            if strategy in failed:continue
            try:
              values,timing=measure(lambda:score(profile,n,strategy))
              delta=compare(baseline,values)
              assert delta['max_abs_logit_delta']<=1e-4 and delta['decision_agreement']==n,delta
              if strategy=='cache_batch':assert all(v['num_cached_tokens']>0 for v in values)
              row=dict(profile=profile,n=n,strategy=strategy,repeat=rep,status='ok',**timing,comparison=delta,results=values)
            except torch.cuda.OutOfMemoryError as exc:
              failed.add(strategy)
              row=dict(profile=profile,n=n,strategy=strategy,repeat=rep,status='oom',error=str(exc))
              gc.collect();torch.cuda.empty_cache()
            rows.append(row);save(a.out/'raw.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k not in ('results','error')}),flush=True)
        save(a.out/'progress.json',dict(profile=profile,n=n,rows=len(rows)))
    summary=[]
    for profile in profiles:
      for n in config['question_counts']:
        for strategy in ['serial','batch','cache_batch']:
          selected=[r for r in rows if (r['profile'],r['n'],r['strategy'])==(profile,n,strategy)]
          ok=[r for r in selected if r['status']=='ok']
          summary.append(dict(profile=profile,n=n,strategy=strategy,status='ok' if len(ok)==a.repeats else 'incomplete_or_oom',
            measured_repeats=len(ok),mean_seconds=statistics.mean(r['seconds'] for r in ok) if ok else None,
            stdev_seconds=statistics.stdev(r['seconds'] for r in ok) if len(ok)>1 else None,
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in ok)/2**30 if ok else None,
            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in ok)/2**30 if ok else None,
            prompt_tokens=[v['prompt_tokens'] for v in ok[0]['results']] if ok else [],
            cached_tokens=[v['num_cached_tokens'] for v in ok[0]['results']] if ok else [],
            max_logit_delta=max(r['comparison']['max_abs_logit_delta'] for r in ok) if ok else None))
    save(a.out/'summary.json',summary)
    (a.out/'_COMPLETE').write_text('All planned cells attempted; inspect individual status for OOM.\n')

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc();raise
