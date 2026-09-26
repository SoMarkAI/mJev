"""Summarize completed controlled runs; never fill missing timings."""
import json
import argparse
import hashlib
import math
import statistics
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, help='Optional directory for regenerated summary/report; default validates without writing')
args=parser.parse_args()
root=Path(__file__).resolve().parent
rows=[]
environments={name:json.loads((root/name/'environment.json').read_text()) for name in ['results','results-long']}
for key in ['source_sha256','model_revision','model_config_sha256','dependencies','python','gpu','code_commit']:
    assert environments['results'][key]==environments['results-long'][key],key
for name,harness in [('results','run-initial.py'),('results-long','run.py')]:
    config=json.loads((root/name/'config.json').read_text())
    assert config==environments[name]['settings']
    assert hashlib.sha256((root/harness).read_bytes()).hexdigest()==config['harness_sha256']
for key in ['model_id','model_revision','code_commit','questions','mode','numerics','projection','dtype','seed','repeats','warmups','question_counts','media_sha256']:
    assert environments['results']['settings'][key]==environments['results-long']['settings'][key],key
for folder,profile in [('results','short'),('results-long','long')]:
    path=root/folder
    assert (path/'_COMPLETE').exists(),folder
    raw=json.loads((path/'raw.json').read_text())
    rows.extend({**r,'source_run':folder} for r in raw if r['profile']==profile)
# Recompute parity from actual candidate scores, independently of stored deltas.
for row in rows:
    if row['status']!='ok':continue
    ref=next(x for x in rows if x['profile']==row['profile'] and x['n']==row['n'] and x['strategy']=='serial' and x['status']=='ok')
    assert len(row['results'])==len(ref['results'])==row['n']
    for a,b in zip(ref['results'],row['results']):
        assert a['id']==b['id'] and a['prompt_tokens']==b['prompt_tokens']
        assert [(v['label'],v['text'],v['token_id']) for v in a['candidates']]==[(v['label'],v['text'],v['token_id']) for v in b['candidates']]
        assert all(math.isfinite(v[k]) for v in b['candidates'] for k in ['raw_logit','probability'])
        assert all(0<=v['probability']<=1 for v in b['candidates'])
        assert abs(sum(v['probability'] for v in b['candidates'])-1)<1e-5
        assert abs(b['probability_sum']-sum(v['probability'] for v in b['candidates']))<1e-8
        assert max(abs(x['raw_logit']-y['raw_logit']) for x,y in zip(a['candidates'],b['candidates']))<=1e-4
        assert a['decision']['label']==b['decision']['label']==max(b['candidates'],key=lambda x:x['raw_logit'])['label']
        assert (b['num_cached_tokens']>0)==(row['strategy']=='cache_batch')
summary=[]
for profile in ['short','long']:
  for n in [1,3,8,16]:
    item={'profile':profile,'questions':n,'strategies':{}}
    for strategy in ['serial','batch','cache_batch']:
      r=[v for v in rows if (v['profile'],v['n'],v['strategy'])==(profile,n,strategy)]
      good=[v for v in r if v['status']=='ok']
      assert len(good)==5 or any(v['status']=='oom' for v in r),(profile,n,strategy)
      if len(good)!=5:
        item['strategies'][strategy]={'status':'oom_or_incomplete','successful_repeats':len(good)}
        continue
      assert sorted(v['repeat'] for v in good)==list(range(5))
      assert all(v['comparison']['decision_agreement']==n for v in good)
      results=good[0]['results']
      item['strategies'][strategy]={
        'status':'ok','seconds':[v['seconds'] for v in good],
        'mean_seconds':statistics.mean(v['seconds'] for v in good),
        'stdev_seconds':statistics.stdev(v['seconds'] for v in good),
        'peak_allocated_gib':max(v['peak_allocated_bytes'] for v in good)/2**30,
        'peak_reserved_gib':max(v['peak_reserved_bytes'] for v in good)/2**30,
        'max_abs_logit_delta':max(v['comparison']['max_abs_logit_delta'] for v in good),
        'prompt_tokens':[v['prompt_tokens'] for v in results],
        'prefix_tokens':results[0]['num_cached_tokens'],
      }
    b=item['strategies']['batch'];c=item['strategies']['cache_batch']
    if b['status']==c['status']=='ok':
      item['cache_vs_batch_speedup']=b['mean_seconds']/c['mean_seconds']
      item['latency_reduction_percent']=(1-c['mean_seconds']/b['mean_seconds'])*100
    summary.append(item)
def check_summary(actual, expected, path='summary'):
    if isinstance(expected, dict):
        assert actual.keys()==expected.keys(), path
        for key in expected:
            check_summary(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual)==len(expected), path
        for i,(a,b) in enumerate(zip(actual, expected)):
            check_summary(a,b,path+'/'+str(i))
    elif isinstance(expected, float):
        # Python versions may differ by one ULP in statistics.stdev.
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-15), path
    else:
        assert actual==expected, path

check_summary(summary,json.loads((root/'combined-summary.json').read_text()))
lines=['# HF prefix KV reuse: controlled performance probe',
'', f"Model: {environments['results']['settings']['model_id']}; revision `{environments['results']['model_revision']}`.",
f"Code: `{environments['results']['code_commit']}`, exact archive; experiment harness separately hashed.",
f"GPU: {environments['results']['gpu']}; Transformers {environments['results']['dependencies']['transformers']}; PyTorch {environments['results']['dependencies']['torch']}.",
'Launcher binds GPU 0 and sets OMP/MKL/OpenBLAS threads to 4. No other GPU compute process was observed at checks; private operational snapshots are not distributed and no continuous exclusivity guarantee is claimed.',
'BF16 weights, stable numerics (FP32 text operations), causal attention, full LM-head projection. No model or inference changes.',
'', '## Scope and timing',
'One six-second project-owned animation. Short and long mean actual model input size, controlled through native video sampling and resolution, not video duration. The 16 questions are fixed task prompts, not dataset ground truth; this is not an accuracy evaluation.',
'Two warmups per cell; five measured repeats, shuffled strategy order per repeat. Strategies execute sequentially; both runs use GPU 0. Synchronized end-to-end score_many time includes media decoding/processing, packing, scoring and fresh prefix prefill. Model load, GC and allocator clearing are excluded. PyTorch allocator is cleared before every measured call, so these values are not directly interchangeable with earlier warm-allocator README figures.',
'Serial means sequential forwards within one score_many call, which shares decoded media; it is not N independent end-to-end API requests. Baseline batch uses the existing stable implementation, including within-forward vision feature deduplication. Cache batches clone/repeat KV; this is not zero-copy shared storage.',
'Peak allocated/reserved include model weights and are PyTorch allocator statistics, not whole-device usage.',
'', '## Results',
'Latency is mean ± sample SD in seconds. Memory columns show maximum peak allocated GiB over five repeats.',
'', '| Input | Questions | Serial s | Batch s | Cache batch s | Cache vs batch | Serial / batch / cache GiB |',
'|---|---:|---:|---:|---:|---:|---:|']
for item in summary:
    vs=item['strategies']
    def t(k):
      x=vs[k]
      return f"{x['mean_seconds']:.3f} ± {x['stdev_seconds']:.3f}" if x['status']=='ok' else x['status']
    memory=' / '.join(f"{vs[k]['peak_allocated_gib']:.2f}" if vs[k]['status']=='ok' else 'OOM' for k in ['serial','batch','cache_batch'])
    ratio=f"{item['cache_vs_batch_speedup']:.2f}×" if 'cache_vs_batch_speedup' in item else 'N/A'
    lines.append(f"| {item['profile']} | {item['questions']} | {t('serial')} | {t('batch')} | {t('cache_batch')} | {ratio} | {memory} |")
lines += ['', '## Input lengths and numerical checks']
for profile in ['short','long']:
    good=[v for x in summary if x['profile']==profile for v in x['strategies'].values() if v['status']=='ok']
    tokens=[n for v in good for n in v['prompt_tokens']]
    prefixes=sorted({v['prefix_tokens'] for v in good if v['prefix_tokens']})
    delta=max(v['max_abs_logit_delta'] for v in good)
    lines.append(f'- {profile}: full prompt {min(tokens)}–{max(tokens)} tokens; cached prefix {prefixes}; max logit delta against same-input serial reference {delta}.')
lines += ['', 'All reported successful runs passed finite-score, within-candidate probability sum and same-input decision checks. Different sampling profiles are not required to produce equal answers.',
'', '## Limits',
'No general threshold, accuracy improvement, native BF16 speed, vLLM speed, Qwen3-Omni-30B speed or audio performance is established. The initial intermediate 178-token-prefix run is retained in results/ but excluded from the short/long table; the corrected long configuration and harness are stored separately. Use raw timings rather than rounding to decide near ties.',
'', 'Raw evidence: results/{config,environment,raw,summary}.json and results-long equivalents; combined-summary.json contains the selected rows and exact timings.']
if args.out:
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.out/'combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('\n'.join(lines))
