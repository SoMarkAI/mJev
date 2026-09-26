"""Strictly paired score/latency report; never mix synthetic Infinity with real labels."""
import argparse
import json
import statistics
import hashlib
import math
from mjev.inputs import candidate_labels
from mjev.benchmark.artifacts import atomic_json, atomic_text
from pathlib import Path
from collections import defaultdict
from mjev.benchmark.evaluate import score_row,aggregate

def percentile(values,q):
    x=sorted(values);pos=(len(x)-1)*q;lo=int(pos);hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)

def load_frozen_manifest(root):
    manifest=json.loads((root/'manifest.json').read_text())
    frozen={}
    if not manifest['modes'] or len(set(manifest['modes'])) != len(manifest['modes']):
        raise ValueError('Empty or duplicate modes')
    for group in manifest['groups']:
        for question in group['questions']:
            if question['id'] in frozen:
                raise ValueError('Duplicate frozen question ID')
            if question['label'] not in candidate_labels(len(question['candidates'])):
                raise ValueError('Invalid frozen answer label')
            frozen[question['id']]={**question, **{key:group[key] for key in ('dataset','reference_type','modality')}}
    if not frozen:
        raise ValueError('Empty frozen manifest')
    return manifest, frozen

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    # A previous successful report must never authorize a failed new attempt.
    (a.out/'_SUCCESS').unlink(missing_ok=True)
    manifest, frozen = load_frozen_manifest(a.out)
    backend_rows={};reports={};rejections={}
    for backend in ('hf','vllm'):
        base=a.out/backend
        if not (base/'_COMPLETE').exists() or (base/'ERROR.json').exists():raise RuntimeError(f'{backend} incomplete or failed')
        rows=[json.loads(x) for x in (base/'raw.jsonl').read_text().splitlines()]
        keyed={(r['id'],r['mode']):r for r in rows}
        if len(keyed)!=len(rows):raise ValueError('Duplicate score rows')
        rejected=json.loads((base/'rejected.json').read_text());rejections[backend]=rejected
        rejected_ids=[row['id'] for row in rejected]
        if len(set(rejected_ids)) != len(rejected_ids) or set(rejected_ids)-set(frozen):
            raise ValueError('Invalid rejection IDs')
        expected={(i,mode) for i in frozen if i not in set(rejected_ids) for mode in manifest['modes']}
        if set(keyed)!=expected:raise ValueError('Incomplete per-mode coverage')
        backend_rows[backend]=keyed
        buckets=defaultdict(list)
        for row in rows:
            q=frozen[row['id']];result=row['result'];labels={c['label']:c['text'] for c in result['candidates']}
            if list(labels) != candidate_labels(len(q['candidates'])) or list(labels.values()) != q['candidates'] or len(labels) != len(result['candidates']):
                raise ValueError('Candidate mapping differs from frozen manifest')
            for field, expected_value in [('dataset',q['dataset']),('modality',q['modality']),('reference_type',q['reference_type']),('reference_label',q['label'])]:
                if row.get(field) != expected_value:
                    raise ValueError(f'{field} differs from frozen manifest for {row["id"]}')
            if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                raise ValueError('Invalid latency')
            metric=score_row({'label':q['label'],'candidates':labels},{'logits':{c['label']:c['raw_logit'] for c in result['candidates']},'prediction':result['decision']['label']})
            for group in ('dataset:'+q['dataset'],'modality:'+q['modality']):buckets[(group,row['mode'])].append((row,metric))
        summary={}
        for (group,mode),items in buckets.items():
            times=[r['seconds'] for r,s in items];metrics=aggregate([s for r,s in items],15);metrics['correct']=sum(s['correct'] for r,s in items)
            reference_types={frozen[r['id']]['reference_type'] for r,s in items}
            if len(reference_types) != 1:
                raise ValueError('Cannot mix reference types in one metric bucket')
            synthetic=reference_types == {'synthetic_proxy'}
            if synthetic:metrics['generated_reference_agreement']=metrics.pop('accuracy')
            summary[group+'|'+mode]={'metric_type':'generated_unreviewed_reference_agreement' if synthetic else 'original_label_accuracy','metrics':metrics,'latency_seconds':{'count':len(times),'mean':statistics.mean(times),'median':statistics.median(times),'p95':percentile(times,.95),'sum':sum(times)},'serial_scoring_questions_per_second':len(times)/sum(times),'ties':sum(len(r['result']['decision']['tied_labels'])>1 for r,s in items)}
        reports[backend]=summary
    if set(backend_rows['hf'])!=set(backend_rows['vllm']):raise RuntimeError('Backend coverage differs; do not report a paired comparison')
    pair_groups=defaultdict(list)
    for key,h in backend_rows['hf'].items():
        v=backend_rows['vllm'][key]
        if h['result']['prompt_token_ids_sha256']!=v['result']['prompt_token_ids_sha256']:raise RuntimeError('Prompt IDs differ')
        pair_groups[(frozen[key[0]]['dataset'],key[1])].append((h,v))
    paired={}
    for (dataset,mode),pairs in pair_groups.items():
        hf_only = vllm_only = same = 0
        reference_types = set()
        for h, v in pairs:
            question = frozen[h['id']]
            answer = question['label']
            h_label = h['result']['decision']['label']
            v_label = v['result']['decision']['label']
            same += h_label == v_label
            hf_only += h_label == answer and v_label != answer
            vllm_only += v_label == answer and h_label != answer
            reference_types.add(question['reference_type'])
        if len(reference_types) != 1:
            raise ValueError('Cannot mix paired reference types')
        paired[dataset+'|'+mode] = {
            'count':len(pairs), 'same_answer':same,
            'hf_only_correct':hf_only, 'vllm_only_correct':vllm_only,
            'reference_type':reference_types.pop(),
        }
    result={'manifest_sha256':hashlib.sha256((a.out/'manifest.json').read_bytes()).hexdigest(),'configuration_evidence':{b:json.loads((a.out/b/'run.json').read_text()) if (a.out/b/'run.json').exists() else {'status':'unverified_historical_configuration','numerics':None,'model_revision':None,'code_commit':None} for b in backend_rows},'reports':reports,'run_timings':{b:json.loads((a.out/b/'run_timing.json').read_text()) for b in backend_rows},'paired':paired,'rejections':rejections,'frozen_questions':len(frozen),'paired_questions':len(backend_rows['hf'])//len(manifest['modes']),'concurrency':1,'kv_cache':'disabled both backends','timing_boundary':manifest['timing_boundary'],'limitations':['Infinity references are generated/unreviewed, not human GT','One timed observation per question/mode; not production P95 or concurrent throughput','Common decode, redundant verification preflight, warmup and startup are outside timed scoring','vLLM TP4 vs HF layer sharding; native kernels and implementation differ','No KV performance claim; HF prefix reuse disabled','Only common accepted 4000-token inputs compared; explicit rejected rows retained'],'result_hashes':{b:hashlib.sha256((a.out/b/'raw.jsonl').read_bytes()).hexdigest() for b in backend_rows}}
    atomic_json(a.out/'report.json', result)
    lines=['# HF vs vLLM paired evaluation','', '| Backend | Data | Mode | N | Metric | Mean seconds/question | P95 seconds/question |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for b,report in reports.items():
        for key,r in sorted(report.items()):
            if not key.startswith('dataset:'):continue
            name,mode=key.split('|');m=r['metrics'];value=m.get('accuracy',m.get('generated_reference_agreement'));t=r['latency_seconds']
            lines.append(f'| {b} | {name[8:]} | {mode} | {m["count"]} | {value:.2%} | {t["mean"]:.3f} | {t["p95"]:.3f} |')
    lines+=['','Infinity metric is generated-reference agreement, not audited accuracy. Other rows use original dataset references.','Concurrent load and prefix-cache speedups are not measured. See report.json for exclusions, NLL/Brier/ECE and paired changes.']
    atomic_text(a.out/'REPORT.md', '\n'.join(lines)+'\n')
    atomic_text(a.out/'_SUCCESS', 'Both backend outputs complete, paired coverage and input hashes match.\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
