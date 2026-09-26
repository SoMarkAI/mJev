"""Frozen input comparison: HF vs pooling vLLM, no KV reuse, sequential requests.

Infinity labels are generated/unreviewed proxies. AV labels are original MCQ or
unanimous native binary references. No reference is passed to model scoring.
"""
import argparse,asyncio,gc,hashlib,json,os,time,statistics
from pathlib import Path
from mjev.benchmark.provenance import evaluation_arguments, capture, check_numerics

def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def freeze(args):
    from evaluate_service import prepare
    args.out.mkdir(parents=True,exist_ok=False);groups=[]
    image_rows=[json.loads(l) for l in (args.infinity/'eval.jsonl').read_text().splitlines()]
    by_image={}
    for q in image_rows:
        by_image.setdefault(q['image'],[]).append({'id':'infinity:'+q['id'],'question':q['question'],'candidates':q['candidates'],'label':q['reference_label']})
    for name,qs in by_image.items():groups.append({'dataset':'infinity','reference_type':'synthetic_proxy','modality':'image','media_path':str(args.infinity/name),'questions':qs})
    av,excluded=prepare(args.root)
    for g in av:
        groups.append({'dataset':g['questions'][0]['dataset'],'reference_type':'original_dataset','modality':g['modality'],'media_path':str(args.root/g['media_path']),'questions':[{'id':q['id'],'question':q['question'],'candidates':list(q['candidates'].values()),'label':q['label']} for q in g['questions']]})
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.infinity/'eval.jsonl',args.infinity/'manifest.json',args.root/'grouped.jsonl']}
    media_hashes={g['media_path']:hashlib.sha256(Path(g['media_path']).read_bytes()).hexdigest() for g in groups}
    save(args.out/'manifest.json',{'groups':groups,'excluded':excluded,'source_hashes':source_hashes,'media_hashes':media_hashes,'modes':['causal','isolated'],'context':'Inspect the supplied media carefully.','video_options':{'fps':1,'min_pixels':3136,'max_pixels':50176,'max_frames':32},'image_options':{'min_pixels':3136,'max_pixels':262144},'full_input_limit':4000,'cache_policy':'no prefix KV; no vLLM multimodal processor cache; common decoded media reused per group','timing_boundary':'in-process per question from template/processor through synchronized model result, excludes common media decode and model startup; both separately recorded','concurrency':1,'hf_projection':'full','seed':37})
    print('FROZEN',len(groups),sum(len(g['questions']) for g in groups),'excluded',len(excluded),flush=True)

def decode(builder,g,manifest):
    from qwen_omni_utils import process_mm_info
    q=g['questions'][0];spec=builder.build(g['media_path'],manifest['context'],q['question'],q['candidates'],modality=g['modality'],video_options=manifest['video_options'] if g['modality'] in ('video','audio_video') else None)
    if g['modality']=='image':spec['messages'][0]['content'][0].update(manifest['image_options'])
    decoded=process_mm_info(spec['messages'],use_audio_in_video=g['modality']=='audio_video',return_video_kwargs=True,image_patch_size=builder.processor.image_processor.patch_size)
    audios,images,videos,kw=decoded
    if audios is not None:
        import numpy as np
        hop=builder.processor.feature_extractor.hop_length
        audios=[np.pad(audio,(0,(-len(audio))%hop)) for audio in audios]
    return audios,images,videos,kw

def vllm_prepared(builder,g,q,manifest,decoded,mode):
    from vllm import PoolingParams
    spec=builder.build(g['media_path'],manifest['context'],q['question'],q['candidates'],modality=g['modality'],video_options=manifest['video_options'] if g['modality'] in ('video','audio_video') else None)
    audios,images,videos,kw=decoded;kw=dict(kw)
    if isinstance(kw.get('fps'),list):
        rates=kw.pop('fps')
        if rates:kw['fps']=float(rates[0])
    if videos is not None:kw['videos_kwargs']={'do_resize':False}
    if images is not None:kw['images_kwargs']={'do_resize':False}
    prompt={'prompt':spec['text'],'multi_modal_data':{k:v for k,v in [('audio',audios),('image',images),('video',videos)] if v is not None},'mm_processor_kwargs':{'use_audio_in_video':g['modality']=='audio_video',**kw}}
    params=PoolingParams(task='classify',use_activation=False,skip_reading_prefix_cache=True,extra_kwargs={'mjev':{'raw_ids':spec['raw_ids'],'spans':spec['spans'],'label_ids':spec['label_ids'],'mode':mode,'debug':False}})
    return prompt,params,spec['labels'],spec['label_ids']

async def run(args):
    import torch
    from mjev.hf import HFMJevEngine
    from mjev.prompt import PromptBuilder
    torch.manual_seed(37);m=json.loads((args.out/'manifest.json').read_text());dest=args.out/args.backend;dest.mkdir(exist_ok=False)
    save(dest/'run.json',capture({'backend':args.backend,'numerics':args.numerics,'dtype':'bfloat16','hf_device_map':'auto','hf_max_memory_gib_per_gpu':20,'vllm_tp':4,'vllm_max_model_len':4096,'mm_processor_cache_gb':0,'frozen_manifest':m},args.model_revision,args.model))
    def sync():
        for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
    started=time.perf_counter();engine=None
    try:
        for i in range(4):
            x=torch.randn(8,8,device=f'cuda:{i}');assert torch.isfinite(x@x).all()
        if args.backend=='hf':
            engine=HFMJevEngine(args.model,max_memory={i:'20GiB' for i in range(4)},numerics=args.numerics,dtype='bfloat16',max_input_tokens=4000);builder=engine.builder
            save(dest/'loading_info.json',{'loading_info':engine.loading_info,'device_map':engine.model.hf_device_map})
        else:
            from mjev.engine import MJevEngine
            engine=MJevEngine(args.model,enable_av=True,enable_prefix_caching=False,max_model_len=4096,mm_processor_cache_gb=0);builder=PromptBuilder(args.model)
        sync();save(dest/'environment.json',{'startup_seconds':time.perf_counter()-started,'backend':args.backend,'torch':torch.__version__,'gpu_names':[torch.cuda.get_device_name(i) for i in range(4)]})
        hf_reference=HFMJevEngine.__new__(HFMJevEngine);hf_reference.builder=builder;hf_reference.max_input_tokens=4000
        for source,expected in m['source_hashes'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Source changed: {source}')
        rows=[];rejects=[];decode_times=[];warmed=set();preflights={};verified_media=set()
        # Smoke selects one group of each dataset, full run processes the frozen manifest.
        groups=m['groups']
        if args.smoke:
            chosen={}
            for g in groups:chosen.setdefault(g['dataset'],g)
            groups=list(chosen.values())
        wall_start=time.perf_counter()
        for gi,g in enumerate(groups):
            if g['media_path'] not in verified_media:
                if hashlib.sha256(Path(g['media_path']).read_bytes()).hexdigest()!=m['media_hashes'][g['media_path']]:raise RuntimeError('Media content changed')
                verified_media.add(g['media_path'])
            start=time.perf_counter()
            try:decoded=decode(builder,g,m)
            except Exception as e:
                for q in g['questions']:rejects.append({'id':q['id'],'stage':'common_decode','error':repr(e)})
                save(dest/'rejected.json',rejects);continue
            decode_times.append({'media_path':g['media_path'],'seconds':time.perf_counter()-start})
            for qi,q in enumerate(g['questions']):
                options=m['video_options'] if g['modality'] in ('video','audio_video') else None
                try:
                    batch,spec,_=hf_reference.prepare(g['media_path'],q['question'],q['candidates'],modality=g['modality'],context=m['context'],video_options=options,decoded=decoded)
                except Exception as e:
                    rejects.append({'id':q['id'],'stage':'common_processor','error':repr(e)});save(dest/'rejected.json',rejects);continue
                expected=digest(batch['input_ids'][0].tolist());preflights[q['id']]={'tokens':spec['prompt_tokens'],'token_hash':expected,'label_ids':spec['label_ids']}
                modes=m['modes'] if (gi+qi)%2==0 else list(reversed(m['modes']))
                for mode in modes:
                    async def score():
                        if args.backend=='hf':
                            b,s,_=engine.prepare(g['media_path'],q['question'],q['candidates'],modality=g['modality'],context=m['context'],video_options=options,decoded=decoded)
                            result=engine.score_prepared(b,s,mode=mode,projection='full');result['prompt_token_ids_sha256']=digest(b['input_ids'][0].tolist());return result
                        prepared=vllm_prepared(builder,g,q,m,decoded,mode)
                        return await engine.score_prepared(prepared,q['question'],q['candidates'],mode=mode)
                    if (g['modality'],mode) not in warmed:
                        await score();sync();warmed.add((g['modality'],mode))
                    sync();start=time.perf_counter();result=await score();sync();elapsed=time.perf_counter()-start
                    if result['prompt_token_ids_sha256']!=expected:raise RuntimeError(f'Backend input token mismatch for {q["id"]}; expected {spec["prompt_tokens"]}, got {result["prompt_tokens"]}')
                    if result['num_cached_tokens']!=0:raise RuntimeError('Unexpected KV cache reuse')
                    if abs(result['probability_sum']-1)>1e-6:raise RuntimeError('Probability normalization failure')
                    row={'id':q['id'],'dataset':g['dataset'],'reference_type':g['reference_type'],'modality':g['modality'],'reference_label':q['label'],'backend':args.backend,'mode':mode,'seconds':elapsed,'result':result}
                    rows.append(row)
                    with (dest/'raw.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                del batch
            save(dest/'progress.json',{'groups_done':gi+1,'groups_total':len(groups),'scored_rows':len(rows),'rejected_questions':len(rejects)})
            save(dest/'preflight.json',preflights);save(dest/'decode_times.json',decode_times)
            if gi%10==0:print('PROGRESS',args.backend,gi+1,len(groups),len(rows),len(rejects),flush=True)
            del decoded;gc.collect()
        save(dest/'run_timing.json',{'evaluation_wall_seconds_including_verification_decode_preflight_warmup':time.perf_counter()-wall_start,'scored_rows':len(rows),'decode_seconds':sum(x['seconds'] for x in decode_times)})
        save(dest/'rejected.json',rejects)
        (dest/'_COMPLETE').write_text('All frozen selected groups attempted; inspect explicit exclusions before comparison.\n')
        print('COMPLETE',args.backend,len(rows),'rejected',len(rejects),flush=True)
    except Exception as e:
        save(dest/'ERROR.json',{'error':repr(e)});raise
    finally:
        if args.backend=='vllm' and engine is not None:engine.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='/model');p.add_argument('--root',type=Path);p.add_argument('--infinity',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=['hf','vllm']);p.add_argument('--freeze',action='store_true');p.add_argument('--smoke',action='store_true');
    # Freezing data has no model configuration; execution does.
    evaluation_arguments(p, required=False)
    a=p.parse_args()
    if not a.freeze:
        if not a.backend or not a.numerics or not a.model_revision:
            p.error('Execution requires --backend, --numerics and --model-revision')
        check_numerics(a.backend, a.numerics)
    if a.freeze:freeze(a)
    else:asyncio.run(run(a))
if __name__=='__main__':main()
