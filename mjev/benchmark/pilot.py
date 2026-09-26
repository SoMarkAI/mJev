"""Small AV end-to-end pilot; same media and raw-head scoring in both modes."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from .dataset import MJevDataset
from .download import atomic_json
from .evaluate import evaluate
from .provenance import evaluation_arguments, capture, check_numerics

VIDEO_OPTIONS={'nframes':8,'min_pixels':8192,'max_pixels':65536}
CONTEXT='Inspect the supplied media carefully.'


def select(dataset, root):
    info={x['media_path']:x for x in json.loads((root/'media_index.json').read_text())}
    flagged={x['media_path'] for x in json.loads((root/'duration_warnings.json').read_text())}
    eligible=[r for r in dataset if info[r['media_path']]['duration_seconds']<=30 and r['media_path'] not in flagged]
    rank=lambda r:hashlib.sha256(('av-pilot-v1:'+r['id']).encode()).hexdigest()
    chosen=[]
    for task in ['sound','music','speech']:
        rows=[r for r in eligible if r['dataset']=='mmau' and r['task']==task]
        if not rows:raise ValueError('No eligible audio task sample: '+task)
        chosen.append(min(rows,key=rank))
    mme=[r for r in dataset if r['dataset']=='video_mme_v2' and r['media_path'] not in flagged]
    videos={r['media_path'] for r in mme}
    groups=[[r for r in mme if r['media_path']==p] for p in videos]
    group=min((g for g in groups if len(g)==4),key=lambda g:(info[g[0]['media_path']]['duration_seconds'],min(rank(r) for r in g)))
    chosen.extend(sorted(group,key=lambda r:r['id']))
    rows=sorted((r for r in eligible if r['dataset']=='mmou'),key=rank)
    seen=set()
    for row in rows:
        if row['media_path'] not in seen:
            seen.add(row['media_path']);chosen.append(row)
        if len(seen)==3:break
    assert len(chosen)==10
    return chosen,info


def prepare(protocol, ds, row):
    labels=list(row['candidates'])
    if labels!=protocol.labels(len(labels)):raise ValueError('Source labels do not match engine labels')
    return protocol.build(str(ds.media_path(row)),CONTEXT,row['question'],list(row['candidates'].values()),
                          modality=row['modality'],video_options=VIDEO_OPTIONS if row['modality']=='audio_video' else None)


def preflight(protocol, ds, chosen, output):
    reports=[]
    for row in chosen:
        start=time.perf_counter();p=prepare(protocol,ds,row);prompt,params,labels,ids=p
        mm=prompt['multi_modal_data'];kwargs=prompt.get('mm_processor_kwargs',{})
        batch=protocol.processor(text=prompt['prompt'],audio=mm.get('audio'),images=mm.get('image'),
                                 videos=mm.get('video'),return_tensors='pt',padding=True,**kwargs)
        token_count=batch['input_ids'].shape[-1]
        if token_count>8192:raise ValueError(f'{row["id"]}: {token_count} tokens exceeds pilot budget')
        raw=params.extra_kwargs['mjev']['raw_ids'];spans=params.extra_kwargs['mjev']['spans']
        expanded=batch['input_ids'][0].tolist();delta=len(expanded)-len(raw)
        if expanded[spans[0][0]+delta:]!=raw[spans[0][0]:]:
            raise ValueError('Official AV processor changed candidate suffix')
        reports.append({'id':row['id'],'modality':row['modality'],'tokens':token_count,
                        'label_token_ids':ids,'preprocess_seconds':time.perf_counter()-start,
                        'fps':kwargs.get('fps'),'candidate_start':spans[0][0]+delta})
    atomic_json(output/'preflight.json',reports)
    return reports


async def infer(args,ds,chosen,out):
    import torch
    from ..engine import MJevEngine
    from .adapters import from_mjev_result
    engine=None;preds={'isolated':[],'causal':[]};outputs=[];timings={'isolated':[],'causal':[]}
    mechanism={}
    try:
        engine=MJevEngine(args.model,enable_av=True)
        for i,row in enumerate(chosen):
            prepared=prepare(engine.protocol,ds,row)
            for mode in (['isolated','causal'] if i%2==0 else ['causal','isolated']):
                start=time.perf_counter()
                result=await engine.score_prepared(prepared,row['question'],list(row['candidates'].values()),mode=mode)
                elapsed=time.perf_counter()-start
                assert abs(result['probability_sum']-1)<1e-6
                preds[mode].append(from_mjev_result(ds.model_input(row),result));timings[mode].append(elapsed)
                outputs.append({'id':row['id'],'seconds':elapsed,**result})
                atomic_json(out/'outputs.json',outputs)
                atomic_json(out/'predictions_by_mode.json',preds)
                print('SCORED',row['id'],mode,result['decision']['label'],round(elapsed,3),flush=True)
        # Real AV inputs, synthetic neutral options: only a mechanism test, never accuracy data.
        for row in [chosen[0],chosen[3]]:
            candidates=['The scene is red.','The scene is blue.','The scene is green.']
            changed=['The scene is tan.',*candidates[1:]]
            options=VIDEO_OPTIONS if row['modality']=='audio_video' else None
            build=lambda c:engine.protocol.build(str(ds.media_path(row)),CONTEXT,'Which description fits?',c,
                                                 modality=row['modality'],video_options=options)
            a,b=build(candidates),build(changed)
            assert a[1].extra_kwargs['mjev']['spans']==b[1].extra_kwargs['mjev']['spans']
            results=[]
            for mode in ['isolated','causal']:
                for prepared,c in [(a,candidates),(b,changed)]:
                    results.append(await engine.score_prepared(prepared,'Which description fits?',c,mode=mode,debug=True))
            h=lambda r:torch.tensor(r['candidate_hidden'])
            iso_delta=float((h(results[0])[1:]-h(results[1])[1:]).abs().max())
            causal_delta=float((h(results[2])[1:]-h(results[3])[1:]).abs().max())
            assert iso_delta<1e-3,iso_delta
            assert causal_delta>1e-3,causal_delta
            warm1=await engine.score_prepared(a,'Which description fits?',candidates)
            warm2=await engine.score_prepared(a,'Which description fits?',candidates)
            assert warm2['num_cached_tokens']>0
            delta=max(abs(x['raw_logit']-y['raw_logit']) for x,y in zip(warm1['candidates'],warm2['candidates']))
            assert delta<0.5,delta
            shuffled=await engine.score_media(str(ds.media_path(row)),'Which description fits?',candidates[::-1],
                modality=row['modality'],context=CONTEXT,video_options=options)
            assert [x['text'] for x in shuffled['candidates']]==candidates[::-1]
            mechanism[row['modality']]={'isolated_hidden_delta':iso_delta,'causal_control_delta':causal_delta,
                'warm_cached_tokens':warm2['num_cached_tokens'],'cache_logit_delta':delta,'shuffle_mapping':'passed'}
            atomic_json(out/'mechanism.json',mechanism)
        # Three questions sharing the same video prefix, independent vLLM requests.
        group=chosen[3:6]
        prepared=[prepare(engine.protocol,ds,r) for r in group]
        concurrent=await asyncio.gather(*(engine.score_prepared(p,r['question'],list(r['candidates'].values())) for p,r in zip(prepared,group)))
        assert len({r['request_id'] for r in concurrent})==3
        assert all(r['num_cached_tokens']>0 for r in concurrent)
        mechanism['concurrent_video_questions']={'requests':3,'cached_tokens':[r['num_cached_tokens'] for r in concurrent]}
        reports={mode:evaluate(chosen,preds[mode]) for mode in preds}
        report={'passed':True,'questions':len(chosen),'scored_requests':len(outputs),'metrics':reports,
                'mechanism':mechanism,'timing':{m:{'mean_seconds':statistics.mean(v),'samples':len(v)} for m,v in timings.items()},
                'timing_scope':'AsyncLLM encode, excluding media preparation and model load; prefix/encoder caches enabled; alternating mode order',
                'limitations':'Small feasibility pilot, not representative benchmark accuracy; eight sampled frames, full audio; no generate()',
                'video_options':VIDEO_OPTIONS,'short_audio_mmou_max_seconds':30, 'mme_selection':'shortest complete video group'}
        atomic_json(out/'report.json',report);(out/'_SUCCESS').write_text('AV pilot GPU scoring and mechanism checks passed.\n')
    except BaseException as exc:
        atomic_json(out/'failure.json',{'type':type(exc).__name__,'error':str(exc),'completed_scoring_requests':len(outputs)})
        raise
    finally:
        if engine is not None:engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--check-only',action='store_true')
    evaluation_arguments(parser, required=False)
    args=parser.parse_args()
    if not args.check_only:
        if not args.numerics or not args.model_revision:
            parser.error('GPU scoring requires --numerics and --model-revision')
        check_numerics('vllm',args.numerics)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'_SUCCESS').exists():raise ValueError('Completed run exists; choose a new output directory')
    os.environ['MJEV_ENABLE']='1';os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    os.environ['MJEV_CACHE_TRACE_PATH']=str(out/'cache_hashes.jsonl')
    from ..patch import install
    install()
    from ..protocol import Protocol
    ds=MJevDataset(Path(args.root)/'manifest.jsonl');chosen,info=select(ds,Path(args.root))
    selection={'ids':[r['id'] for r in chosen],'source_manifest_sha256':hashlib.sha256(ds.manifest.read_bytes()).hexdigest(),
               'selection':'deterministic SHA256 rank among <=30s MMAU/MMOU files, exclude duration warnings; one per MMAU task, shortest complete MME video group, three MMOU videos',
               'video_options':VIDEO_OPTIONS,'media_sha256':{r['media_path']:info[r['media_path']]['sha256'] for r in chosen}}
    if (out/'selection.json').exists() and json.loads((out/'selection.json').read_text())!=selection:
        raise ValueError('Selection or processing policy changed')
    atomic_json(out/'selection.json',selection)
    atomic_json(out/'run.json',capture({'backend':'historical_av_pilot','check_only':args.check_only,'numerics':args.numerics,'dtype':'bfloat16','projection':'full','max_model_len':8192,'prefix_cache':True,'modes':['causal','isolated'],'context':CONTEXT,'video_options':VIDEO_OPTIONS,'selection':selection},args.model_revision,args.model))
    preflight(Protocol(args.model),ds,chosen,out)
    if args.check_only:print('AV_PREFLIGHT_OK 10 questions, no model inference',flush=True)
    else:asyncio.run(infer(args,ds,chosen,out))

if __name__=='__main__':main()
