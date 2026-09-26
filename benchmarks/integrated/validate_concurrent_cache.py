"""Small real multi-question batch/cache probe; no dataset accuracy claim.

Prepared inputs are identical across strategies. Timings exclude common media
decode/processor work and include packing, KV copies, forward, and readout.
Cold prefix construction is recorded separately and added to cold-group cost.
"""
import argparse,asyncio,copy,gc,hashlib,json,os,time
from collections import Counter
from pathlib import Path
from mjev.benchmark.provenance import evaluation_arguments, capture, check_numerics
import torch
from compare_backends import decode,vllm_prepared,digest,save
from mjev.hf import HFMJevEngine,HFContext
from mjev.prompt import PromptBuilder
from hf_batch_probe import score_batch

def freeze(args):
    source=json.loads(args.source.read_text())
    pf=json.loads((args.source.parent/'vllm/preflight.json').read_text())
    chosen=[]
    for dataset in ['infinity','clotho_aqa','muchomusic','video_mme_v2','mmou']:
        candidates=[]
        for g in source['groups']:
            qs=[q for q in g['questions'] if q['id'] in pf and pf[q['id']]['tokens']<=3500][:3]
            if g['dataset']==dataset and len(qs)>=2:
                candidates.append((max(pf[q['id']]['tokens'] for q in qs),{**g,'questions':qs}))
        candidates.sort(key=lambda x:(x[0],x[1]['media_path']))
        if not candidates:raise ValueError(f'No eligible group for {dataset}')
        for idx in sorted({0,len(candidates)//2}):chosen.append(candidates[idx][1])
    args.out.mkdir(parents=True,exist_ok=False)
    manifest={**source,'groups':chosen,'excluded':[],
        'media_hashes':{g['media_path']:source['media_hashes'][g['media_path']] for g in chosen},
        'source_manifest_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'scope':'Two media per dataset where available: shortest and median maximum input length; 2-3 original questions each. Mechanism pilot, not accuracy benchmark.',
        'repeats':3,'max_concurrency':3,'logit_atol':0.1,'probability_atol':0.02,
        'cache_policy':'Compare bypass, cold automatic reuse and warmed cross-question reuse. HF repeats cloned prefix KV across batch rows; not zero-copy.',
        'timing_boundary':'Prepared-input group scoring wall time, excludes decoding and processor. Includes batch packing and KV cloning. Prefix construction/priming recorded separately.'}
    save(args.out/'manifest.json',manifest)
    print('FROZEN',len(chosen),sum(len(g['questions']) for g in chosen),flush=True)

def sync():
    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)

def compare(a,b,manifest):
    if len(a)!=len(b):raise ValueError('Question count mismatch')
    checks=[]
    for x,y in zip(a,b):
        if [c['text'] for c in x['candidates']]!=[c['text'] for c in y['candidates']]:
            raise ValueError('Candidate mismatch')
        if x['prompt_tokens']!=y['prompt_tokens']:raise ValueError('Length mismatch')
        logit=max(abs(c['raw_logit']-d['raw_logit']) for c,d in zip(x['candidates'],y['candidates']))
        prob=max(abs(c['probability']-d['probability']) for c,d in zip(x['candidates'],y['candidates']))
        same=x['decision']['label']==y['decision']['label']
        checks.append(dict(same_answer=same,max_abs_logit_error=logit,
            max_abs_probability_error=prob,passed=same and logit<=manifest['logit_atol'] and prob<=manifest['probability_atol']))
    return checks

def kv_digest(cache):
    h=hashlib.sha256()
    for layer in cache.past_key_values.layers:
        for tensor in (layer.keys,layer.values):
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def hf_measure(engine,fn):
    calls=Counter();frames=[]
    def event(module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is not None:frames.append(dict(batch_size=int(ids.shape[0]),query_width=int(ids.shape[1])))
    handles=[engine.model.register_forward_pre_hook(event,with_kwargs=True)]
    for key,tower in [('audio',engine.model.audio_tower),('vision',engine.model.visual)]:
        handles.append(tower.register_forward_hook(lambda m,a,o,k=key:calls.update([k])))
    try:
        sync();start=time.perf_counter();value=fn();sync();seconds=time.perf_counter()-start
    finally:
        for h in handles:h.remove()
    return value,dict(group_seconds=seconds,encoder_calls=dict(calls),forward_frames=frames)

async def run(args):
    m=json.loads((args.out/'manifest.json').read_text());dest=args.out/args.backend
    dest.mkdir(exist_ok=False)
    save(dest/'run.json',capture({'backend':args.backend,'numerics':args.numerics,'dtype':'bfloat16','hf_device_map':'auto','hf_max_memory_gib_per_gpu':20,'vllm_tp':4,'vllm_max_model_len':4096,'mm_processor_cache_gb':0,'frozen_manifest':m},args.model_revision,args.model));engine=None
    save(dest/'source_hashes.json',{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(Path(__file__).resolve().parents[2].rglob('*.py')) if '.aris' not in p.parts})
    try:
        torch.manual_seed(37)
        assert torch.cuda.device_count()==4
        for i in range(4):
            x=torch.randn(8,8,device=f'cuda:{i}');assert torch.isfinite(x@x).all()
        start=time.perf_counter()
        if args.backend=='hf':
            engine=HFMJevEngine(args.model,max_memory={i:'20GiB' for i in range(4)},numerics=args.numerics,dtype='bfloat16',max_input_tokens=4000)
            builder=engine.builder
            assert set(map(str,engine.model.hf_device_map.values()))=={'0','1','2','3'}
            save(dest/'loading_info.json',engine.loading_info)
        else:
            from mjev.engine import MJevEngine
            engine=MJevEngine(args.model,enable_av=True,enable_prefix_caching=True,
                             max_model_len=4096,mm_processor_cache_gb=0)
            builder=PromptBuilder(args.model)
        sync();save(dest/'environment.json',dict(startup_seconds=time.perf_counter()-start,
            torch=torch.__version__,gpus=[torch.cuda.get_device_name(i) for i in range(4)],backend=args.backend))
        reference=HFMJevEngine.__new__(HFMJevEngine);reference.builder=builder;reference.max_input_tokens=4000
        trace_path=dest/'batch_trace.jsonl'
        records=[]
        for gi,g in enumerate(m['groups']):
            if hashlib.sha256(Path(g['media_path']).read_bytes()).hexdigest()!=m['media_hashes'][g['media_path']]:
                raise ValueError('Media content changed')
            prep_start=time.perf_counter();decoded=decode(builder,g,m);prepared=[]
            for q in g['questions']:
                b,s,_=reference.prepare(g['media_path'],q['question'],q['candidates'],
                    modality=g['modality'],context=m['context'],decoded=decoded,
                    video_options=m['video_options'] if g['modality']=='audio_video' else None)
                prepared.append((b,s))
            prep_seconds=time.perf_counter()-prep_start
            expected=[digest(b['input_ids'][0].tolist()) for b,s in prepared]
            modes=m['modes'] if gi%2==0 else list(reversed(m['modes']))
            for mode in modes:
                base={'dataset':g['dataset'],'modality':g['modality'],'media_path':g['media_path'],
                    'question_ids':[q['id'] for q in g['questions']], 'mode':mode,
                    'prompt_tokens':[s['prompt_tokens'] for b,s in prepared],
                    'prompt_hashes':expected,'common_decode_processor_seconds':prep_seconds}
                if args.backend=='vllm':
                    vp=[vllm_prepared(builder,g,q,m,decoded,mode) for q in g['questions']]
                    prime_q={'question':'Describe the media.','candidates':['Yes','No']}
                    prime=vllm_prepared(builder,g,prime_q,m,decoded,mode)
                    async def single(p,q,skip):
                        p=copy.deepcopy(p);p[1].skip_reading_prefix_cache=skip
                        begin=time.perf_counter()
                        r=await engine.score_prepared(p,q['question'],q['candidates'],mode=mode)
                        return r,time.perf_counter()-begin
                    async def execute(parallel,skip):
                        if parallel:items=await asyncio.gather(*(single(p,q,skip) for p,q in zip(vp,g['questions'])))
                        else:items=[await single(p,q,skip) for p,q in zip(vp,g['questions'])]
                        return [r for r,t in items],[t for r,t in items]
                    # Warm both scheduling shapes. No warming result enters metrics.
                    await execute(False,True);await execute(True,True)
                    names=['serial_no_cache','parallel_no_cache','parallel_cold_cache','serial_warm_cache','parallel_warm_cache']
                    for rep in range(m['repeats']):
                        variants={}
                        for name in names[rep:]+names[:rep]:
                            assert await engine.engine.reset_prefix_cache(),'Cache reset failed'
                            prime_seconds=0.;prime_result=None
                            if 'warm_cache' in name:
                                sync();begin=time.perf_counter()
                                prime_result,_=await single(prime,prime_q,False)
                                sync();prime_seconds=time.perf_counter()-begin
                            offset=trace_path.stat().st_size if trace_path.exists() else 0
                            sync();begin=time.perf_counter()
                            values,latencies=await execute(name.startswith('parallel'),name.endswith('no_cache'))
                            sync();seconds=time.perf_counter()-begin
                            frames=[]
                            if trace_path.exists():
                                with trace_path.open() as f:
                                    f.seek(offset);frames=[json.loads(l) for l in f if l.strip()]
                            if [r['prompt_token_ids_sha256'] for r in values]!=expected:raise ValueError('Prompt hash mismatch')
                            if name.endswith('no_cache') and any(r['num_cached_tokens'] for r in values):raise ValueError('Bypass reused cache')
                            variants[name]=dict(values=values,group_seconds=seconds,request_seconds=latencies,
                                prime_seconds=prime_seconds,cold_total_seconds=prime_seconds+seconds,
                                prime_result=prime_result,forward_frames=frames,
                                max_scheduled_questions=max([sum(t>0 for t in f['query_tokens']) for f in frames] or [0]))
                        contrasts={n:compare(variants['serial_no_cache']['values'],v['values'],m) for n,v in variants.items() if n!='serial_no_cache'}
                        contrasts['cache_effect_at_parallel']=compare(variants['parallel_no_cache']['values'],variants['parallel_warm_cache']['values'],m)
                        reversed_pairs=list(reversed(list(zip(vp,g['questions']))))
                        reversed_outputs=await asyncio.gather(*(single(p,q,False) for p,q in reversed_pairs))
                        order_checks=compare(variants['parallel_warm_cache']['values'],
                            list(reversed([x[0] for x in reversed_outputs])),m)
                        row={**base,'repeat':rep,'variants':variants,'comparisons':contrasts,'question_order_checks':order_checks,
                            'question_order_values':list(reversed([x[0] for x in reversed_outputs]))}
                        records.append(row)
                        with (dest/'raw.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                else:
                    def serial(cache=None):return [engine.score_prepared(b,s,mode=mode,projection='full',cache=cache) for b,s in prepared]
                    def make_context():
                        b,s=prepared[0];past,positions=engine._prefill(b,s['prefix_length'])
                        return HFContext(engine,g['media_path'],g['modality'],m['context'],None,
                                         decoded,copy.deepcopy(b),past,positions,s['prefix_length'])
                    serial();score_batch(engine,prepared,mode=mode)
                    warm=make_context();score_batch(engine,prepared,mode=mode,cache=warm);del warm
                    for rep in range(m['repeats']):
                        variants={};cache,prime_timing=hf_measure(engine,make_context)
                        before=kv_digest(cache)
                        names=['serial_no_cache','parallel_no_cache','serial_warm_cache','parallel_warm_cache']
                        for name in names[rep:]+names[:rep]:
                            c=cache if 'warm_cache' in name else None
                            if name.startswith('parallel'):fn=lambda:score_batch(engine,prepared,mode=mode,cache=c)[0]
                            else:fn=lambda:serial(c)
                            values,timing=hf_measure(engine,fn)
                            for result,h in zip(values,expected):result['prompt_token_ids_sha256']=h
                            if c is not None and sum(timing['encoder_calls'].values()):raise ValueError('Cached suffix re-encoded media')
                            variants[name]=dict(values=values,**timing,
                                prime_seconds=prime_timing['group_seconds'] if c is not None else 0.,
                                cold_total_seconds=timing['group_seconds']+(prime_timing['group_seconds'] if c is not None else 0.))
                        reversed_values,_=score_batch(engine,list(reversed(prepared)),mode=mode,cache=cache)
                        order_checks=compare(variants['parallel_warm_cache']['values'],list(reversed(reversed_values)),m)
                        if before!=kv_digest(cache):raise ValueError('Base cache mutated across branches')
                        contrasts={n:compare(variants['serial_no_cache']['values'],v['values'],m) for n,v in variants.items() if n!='serial_no_cache'}
                        contrasts['cache_effect_at_parallel']=compare(variants['parallel_no_cache']['values'],variants['parallel_warm_cache']['values'],m)
                        row={**base,'repeat':rep,'variants':variants,'comparisons':contrasts,
                            'prefill_timing':prime_timing,'base_cache_sha256':before,'base_cache_unchanged':True,'question_order_checks':order_checks,
                            'question_order_values':list(reversed(reversed_values))}
                        records.append(row)
                        with (dest/'raw.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                        del cache;gc.collect()
                save(dest/'progress.json',dict(groups_done=gi+1,groups_total=len(m['groups']),mode=mode,records=len(records)))
                print('DONE',args.backend,g['dataset'],gi+1,mode,len(records),flush=True)
            del decoded,prepared;gc.collect()
        checks=[c for r in records for name,cs in r['comparisons'].items() for c in cs]
        order_checks=[c for r in records for c in r.get('question_order_checks',[])]
        summary=dict(question_order_checks=len(order_checks),question_order_passed=all(c['passed'] for c in order_checks),groups=len(m['groups']),questions=sum(len(g['questions']) for g in m['groups']),
            records=len(records),total_checks=len(checks),same_answer_checks=sum(c['same_answer'] for c in checks),
            passing_checks=sum(c['passed'] for c in checks),all_parity_passed=all(c['passed'] for c in checks+order_checks),
            max_abs_logit_error=max(c['max_abs_logit_error'] for c in checks),
            max_abs_probability_error=max(c['max_abs_probability_error'] for c in checks))
        save(dest/'summary.json',summary)
        (dest/'_COMPLETE').write_text('All selected cases executed. Read summary: completion does not imply cache parity passed.\n')
        if summary['all_parity_passed']:(dest/'_PARITY_PASS').write_text('Numerical tolerances and answer equality passed for all recorded comparisons.\n')
    except Exception as e:
        save(dest/'ERROR.json',dict(error=repr(e)));raise
    finally:
        if args.backend=='vllm' and engine is not None:engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='/model');p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=['hf','vllm']);p.add_argument('--freeze',action='store_true');
    # Freezing data has no model configuration; execution does.
    evaluation_arguments(p, required=False)
    a=p.parse_args()
    if not a.freeze:
        if not a.backend or not a.numerics or not a.model_revision:
            p.error('Execution requires --backend, --numerics and --model-revision')
        check_numerics(a.backend, a.numerics)
    if a.freeze:freeze(a)
    else:asyncio.run(run(a))
