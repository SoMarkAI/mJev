"""Grouped, answer-free requests; native Clotho references are never fabricated."""
import argparse,json,hashlib,time,collections,statistics
from pathlib import Path
import requests
from mjev.benchmark.evaluate import evaluate
from mjev.benchmark.provenance import capture, write

def prepare(root):
    groups=[];excluded=[]
    for g in map(json.loads,(root/'grouped.jsonl').read_text().splitlines()):
        rows=[]
        for original in g['questions']:
            q=dict(original)
            if q.get('candidates') is None:
                adapter=q.get('binary_adapter')
                if not adapter or adapter['label'] is None:
                    excluded.append({'id':q['id'],'reason':'native_freeform_or_disagreeing_references'});continue
                q.update(candidates=adapter['candidates'],label=adapter['label'],task='binary_audio_qa_unanimous')
            rows.append(q)
        for offset in range(0,len(rows),6):
            groups.append({'media_path':g['media_path'],'modality':g['modality'],'questions':rows[offset:offset+6]})
    return groups,excluded

def payload(g,mode):
    return {'mode':mode,'use_audio_in_video':g['modality']=='audio_video',
            'media':[{'type':'audio_url' if g['modality']=='audio' else 'video_url',
                      'url':'file:///data/'+g['media_path']}],
            'questions':[{'id':q['id'],'instructions':q['question'],'criteria':q['candidates']} for q in g['questions']]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:17005/v1/mjev/multi_question');p.add_argument('--prepare-only',action='store_true');p.add_argument('--modes',nargs='+',choices=['causal','masked'],default=['causal','masked']);p.add_argument('--limit-per-modality',type=int);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);groups,excluded=prepare(args.root)
    if args.limit_per_modality:
        counts=collections.Counter();limited=[]
        for g in groups:
            mod='audio' if g['modality']=='audio' else 'video'
            if counts[mod]<args.limit_per_modality:limited.append(g);counts[mod]+=1
        groups=limited
    rows=[q for g in groups for q in g['questions']]
    (args.out/'evaluation_manifest.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(args.out/'excluded.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in excluded))
    coverage={'source_questions':sum(len(g['questions']) for g in map(json.loads,(args.root/'grouped.jsonl').read_text().splitlines())),'eligible_questions':len(rows),'requests_per_mode':len(groups),'excluded_native_questions':len(excluded),'datasets':dict(collections.Counter(q['dataset'] for q in rows)),'source_sha256':hashlib.sha256((args.root/'manifest.jsonl').read_bytes()).hexdigest(),'note':'Source token bounds are not revalidated for this service template. Video requests explicitly include native audio extraction. No answer/evidence fields sent to server.'}
    (args.out/'coverage.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage),flush=True)
    if args.prepare_only:return
    evidence=capture({'backend':'experimental_http_tree_kv','scope':'client_only','modes':args.modes,'group_limit':6,'request_timeout_seconds':600,'numerics':None,'server_configuration_status':'unverified_not_attested','server_model_revision':None,'server_dependencies':None,'source_manifest_sha256':coverage['source_sha256']},None)
    write(args.out/'run.json',evidence)
    predictions={m:[] for m in args.modes};times={m:[] for m in args.modes}
    with requests.Session() as session, (args.out/'raw_responses.jsonl').open('w') as log:
        for i,g in enumerate(groups):
            modes=args.modes if i%2==0 else list(reversed(args.modes))
            for mode in modes:
                body=payload(g,mode);start=time.perf_counter()
                try:
                    response=session.post(args.url,json=body,timeout=600);response.raise_for_status();result=response.json();elapsed=time.perf_counter()-start
                    expected={q['id']:q for q in g['questions']};answers=result['answers'];assert {a['id'] for a in answers}==set(expected) and len(answers)==len(expected)
                    for a in answers:
                        assert abs(sum(a['probabilities'].values())-1)<1e-6
                        predictions[mode].append({'id':a['id'],'logits':a['scores'],'prediction':a['answer']})
                    times[mode].append(elapsed)
                    log.write(json.dumps({'group':i,'mode':mode,'request':body,'response':result,'client_seconds':elapsed})+'\n');log.flush()
                    (args.out/(mode+'.predictions.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in predictions[mode]))
                    print(mode,i+1,len(groups),round(elapsed,3),flush=True)
                except Exception as e:
                    (args.out/'ERROR.json').write_text(json.dumps({'group':i,'mode':mode,'error':repr(e),'response':response.text[:4000] if 'response' in locals() else None}));raise
    report={m:{'evaluation':evaluate(rows,predictions[m]),'client_request_seconds':{'median':statistics.median(times[m]),'mean':statistics.mean(times[m]),'count':len(times[m])}} for m in args.modes}
    report['_configuration_evidence']=evidence
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));(args.out/'_SUCCESS').write_text('All eligible questions completed in each requested mode. See coverage for exclusions.\n')
if __name__=='__main__':main()
