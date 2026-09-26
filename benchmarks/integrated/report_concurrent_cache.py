"""Summarize completed concurrency probes without equating hits with correctness."""
import argparse,json,hashlib,statistics
from pathlib import Path
from collections import defaultdict

def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    expected={(g['media_path'],mode,rep) for g in manifest['groups']
              for mode in manifest['modes'] for rep in range(manifest['repeats'])}
    output={'scope':manifest['scope'],'media':len(manifest['groups']),
        'questions':sum(len(g['questions']) for g in manifest['groups']),
        'repeats':manifest['repeats'],'logit_atol':manifest['logit_atol'],
        'probability_atol':manifest['probability_atol'],
        'timing_boundary':manifest['timing_boundary'],'backends':{},'raw_sha256':{},
        'limitations':['Small selected pilot, not dataset accuracy or production load test.',
          'HF score_many supports real batching with batch_size; its default remains 1.',
          'HF cached batch repeats cloned prefix KV per row, not zero-copy sharing.',
          'vLLM uses patched pooling/SDPA, not the HTTP Tree-KV runtime.',
          'vLLM warm priming uses an extra different question; priming cost is recorded.',
          'Common media decode and processor work are excluded from scoring times.',
          'Cache hits, actual batching and numerical acceptance are separate checks.']}
    if not expected:raise ValueError('Empty experiment')
    group_by_path={g['media_path']:g for g in manifest['groups']}
    for backend in ('vllm','hf'):
        dest=root/backend
        if not (dest/'_COMPLETE').exists() or (dest/'ERROR.json').exists():raise ValueError(f'{backend} incomplete')
        raw=(dest/'raw.jsonl').read_bytes();rows=[json.loads(l) for l in raw.splitlines()]
        keys=[(r['media_path'],r['mode'],r['repeat']) for r in rows]
        if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Coverage mismatch')
        output['raw_sha256'][backend]=hashlib.sha256(raw).hexdigest()
        buckets=defaultdict(list);checks=defaultdict(list);bad=[];repeated=defaultdict(list);order=[]
        for row in rows:
            group=group_by_path[row['media_path']];qs=group['questions']
            if row['question_ids']!=[q['id'] for q in qs]:raise ValueError('Question identity mismatch')
            baseline=row['variants']['serial_no_cache']['values']
            if len(baseline)!=len(qs):raise ValueError('Baseline count mismatch')
            for strategy,v in row['variants'].items():
                if len(v['values'])!=len(qs):raise ValueError('Missing score')
                for q,r,h in zip(qs,v['values'],row['prompt_hashes']):
                    if [c['text'] for c in r['candidates']]!=q['candidates']:raise ValueError('Candidates changed')
                    if r['prompt_token_ids_sha256']!=h:raise ValueError('Input hash changed')
                    if abs(r['probability_sum']-1)>1e-6:raise ValueError('Invalid softmax sum')
                    if strategy.endswith('no_cache') and r['num_cached_tokens']!=0:raise ValueError('Unexpected cache hit')
                buckets[(row['mode'],strategy)].append((row,v))
                repeated[(row['media_path'],row['mode'],strategy)].append((row['repeat'],v['values']))
            if 'question_order_values' in row:
                a=row['variants']['parallel_warm_cache']['values'];b=row['question_order_values']
                if len(a)!=len(b) or len(a)!=len(row['question_order_checks']):raise ValueError('Order count mismatch')
                for x,y,c in zip(a,b,row['question_order_checks']):
                    if [v['text'] for v in x['candidates']]!=[v['text'] for v in y['candidates']]:raise ValueError('Order candidate mismatch')
                    le=max(abs(v['raw_logit']-w['raw_logit']) for v,w in zip(x['candidates'],y['candidates']))
                    pe=max(abs(v['probability']-w['probability']) for v,w in zip(x['candidates'],y['candidates']))
                    same=x['decision']['label']==y['decision']['label']
                    actual=dict(same_answer=same,max_abs_logit_error=le,max_abs_probability_error=pe,
                                passed=same and le<=manifest['logit_atol'] and pe<=manifest['probability_atol'])
                    if actual!=c:raise ValueError('Order comparison does not match raw scores')
                    order.append(actual)
            for contrast,cs in row['comparisons'].items():
                left=baseline
                right=row['variants'][contrast]['values'] if contrast!='cache_effect_at_parallel' else row['variants']['parallel_warm_cache']['values']
                if contrast=='cache_effect_at_parallel':left=row['variants']['parallel_no_cache']['values']
                if len(cs)!=len(qs):raise ValueError('Comparison count mismatch')
                recomputed=[]
                for x,y,stored in zip(left,right,cs):
                    logit=max(abs(a['raw_logit']-b['raw_logit']) for a,b in zip(x['candidates'],y['candidates']))
                    prob=max(abs(a['probability']-b['probability']) for a,b in zip(x['candidates'],y['candidates']))
                    same=x['decision']['label']==y['decision']['label']
                    c=dict(same_answer=same,max_abs_logit_error=logit,max_abs_probability_error=prob,
                           passed=same and logit<=manifest['logit_atol'] and prob<=manifest['probability_atol'])
                    if c!=stored:raise ValueError('Stored comparison does not match raw scores')
                    recomputed.append(c)
                cs=recomputed
                checks[(row['mode'],contrast)].extend(cs)
                for q,c in zip(qs,cs):
                    if not c['passed']:bad.append(dict(question_id=q['id'],mode=row['mode'],repeat=row['repeat'],contrast=contrast,**c))
        report={}
        for (mode,strategy),pairs in buckets.items():
            times=[v['group_seconds'] for r,v in pairs]
            values=[x for r,v in pairs for x in v['values']]
            widths=[v.get('max_scheduled_questions',max([f['batch_size'] for f in v['forward_frames']] or [0]))
                    if backend=='hf' else v['max_scheduled_questions'] for r,v in pairs]
            report[mode+'|'+strategy]=dict(groups=len(pairs),scored_questions=len(values),
                group_seconds_mean=statistics.mean(times),group_seconds_median=statistics.median(times),
                prime_seconds_mean=statistics.mean(v['prime_seconds'] for r,v in pairs),
                cold_total_seconds_mean=statistics.mean(v['cold_total_seconds'] for r,v in pairs),
                questions_per_scoring_second=len(values)/sum(times),
                cache_hit_questions=sum(v['num_cached_tokens']>0 for v in values),
                cached_tokens_min=min(v['num_cached_tokens'] for v in values),
                cached_tokens_max=max(v['num_cached_tokens'] for v in values),
                actual_batch_max=max(widths),groups_with_batch_gt1=sum(w>1 for w in widths),
                groups_with_batch3=sum(w==3 for w in widths),
                encoder_calls=sum(sum(v.get('encoder_calls',{}).values()) for r,v in pairs) if backend=='hf' else None)
        parity={mode+'|'+contrast:dict(count=len(cs),same_answer=sum(c['same_answer'] for c in cs),
            passed=sum(c['passed'] for c in cs),all_passed=all(c['passed'] for c in cs),
            max_abs_logit_error=max(c['max_abs_logit_error'] for c in cs),
            max_abs_probability_error=max(c['max_abs_probability_error'] for c in cs))
            for (mode,contrast),cs in checks.items()}
        repeat_checks=defaultdict(list)
        for (media,mode,strategy),runs in repeated.items():
            runs.sort(key=lambda x:x[0]);first=runs[0][1]
            for _,later in runs[1:]:
                for a,b in zip(first,later):
                    repeat_checks[mode+'|'+strategy].append(dict(
                        same_answer=a['decision']['label']==b['decision']['label'],
                        max_abs_logit_error=max(abs(x['raw_logit']-y['raw_logit']) for x,y in zip(a['candidates'],b['candidates']))))
        consistency={k:dict(count=len(cs),same_answer=sum(c['same_answer'] for c in cs),
                       max_abs_logit_error=max(c['max_abs_logit_error'] for c in cs)) for k,cs in repeat_checks.items()}
        output['backends'][backend]=dict(strategies=report,comparisons=parity,failed_checks=bad,repeat_consistency=consistency,
            question_order=dict(count=len(order),passed=sum(c['passed'] for c in order),
                max_abs_logit_error=max((c['max_abs_logit_error'] for c in order),default=None)))
    output['configuration_evidence']={b:json.loads((root/b/'run.json').read_text()) if (root/b/'run.json').exists() else {'status':'unverified_historical_configuration','numerics':None,'model_revision':None,'code_commit':None} for b in ('hf','vllm')}
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    (a.out/'_REPORT_COMPLETE').unlink(missing_ok=True)
    out=summarize(a.out);(a.out/'report.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# Concurrent questions and prefix KV cache probe','',
           f'{out["media"]} media; {out["questions"]} original questions; {out["repeats"]} repeats per mode/strategy.','',
           '| Backend | Mode | Strategy | Mean group s | With cold priming s | Cache hits / scores | Actual batch max |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for b,r in out['backends'].items():
        for name,s in r['strategies'].items():
            mode,strategy=name.split('|')
            lines.append(f'| {b} | {mode} | {strategy} | {s["group_seconds_mean"]:.4f} | {s["cold_total_seconds_mean"]:.4f} | {s["cache_hit_questions"]}/{s["scored_questions"]} | {s["actual_batch_max"]} |')
    lines+=['','## Numerical acceptance','','| Backend | Mode / contrast | Same answer | All tolerances pass | Max logit error | Max probability error |','| --- | --- | ---: | ---: | ---: | ---: |']
    for b,r in out['backends'].items():
        for name,c in r['comparisons'].items():
            lines.append(f'| {b} | {name.replace(chr(124), chr(32)+chr(47)+chr(32))} | {c["same_answer"]}/{c["count"]} | {c["passed"]}/{c["count"]} | {c["max_abs_logit_error"]:.5f} | {c["max_abs_probability_error"]:.5f} |')
    lines+=['','Tolerance: identical answer AND max absolute raw-logit difference <=0.1 AND max probability difference <=0.02.','',out['timing_boundary'],'']+['- '+x for x in out['limitations']]
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.out/'_REPORT_COMPLETE').write_text('Coverage and report assembly complete. This is not a parity-pass marker.\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
