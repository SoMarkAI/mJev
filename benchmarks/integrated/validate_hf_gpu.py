"""Small full-weight HF cache validation; not a dataset accuracy benchmark."""
import argparse,json,time,hashlib,sys,gc
from pathlib import Path
from mjev.benchmark.provenance import evaluation_arguments, capture
import torch
from evaluate_service import prepare
from mjev.hf import HFMJevEngine

def sync():
    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)

def measure(fn):
    sync()
    for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
    start=time.perf_counter();value=fn();sync()
    return value,{'seconds':time.perf_counter()-start,'peak_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}

def compare(a,b):
    x=torch.tensor([c['raw_logit'] for c in a['candidates']]);y=torch.tensor([c['raw_logit'] for c in b['candidates']])
    p=torch.tensor([c['probability'] for c in a['candidates']]);q=torch.tensor([c['probability'] for c in b['candidates']])
    assert a['prompt_tokens']==b['prompt_tokens']
    assert [c['label'] for c in a['candidates']]==[c['label'] for c in b['candidates']]
    for r in (a,b):assert abs(r['probability_sum']-1)<1e-6
    return {'max_abs_logit_error':float((x-y).abs().max()),'max_abs_probability_error':float((p-q).abs().max()),'same_answer':a['decision']['label']==b['decision']['label']}

def checkpoint_witness(engine,model_path):
    """Exact source/runtime equality for two experts in every decoder layer.

    This is a sampled tensor witness, not a checksum of the complete checkpoint.
    """
    from safetensors import safe_open
    root=Path(model_path);index=root/'model.safetensors.index.json'
    weights=json.loads(index.read_text())['weight_map'];rows=[]
    for layer_id,layer in enumerate(engine.model.model.layers):
        experts=layer.mlp.experts
        for expert_id in (0,experts.gate_up_proj.shape[0]-1):
            gate,up=experts.gate_up_proj[expert_id].chunk(2,dim=0)
            for name,actual in [('gate_proj',gate),('up_proj',up),('down_proj',experts.down_proj[expert_id])]:
                key=f'thinker.model.layers.{layer_id}.mlp.experts.{expert_id}.{name}.weight'
                with safe_open(str(root/weights[key]),framework='pt',device='cpu') as f:source=f.get_tensor(key)
                value=actual.detach().cpu()
                if not torch.equal(source.to(value.dtype),value):raise RuntimeError(f'Checkpoint tensor mismatch: {key}')
                digest=hashlib.sha256(source.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                rows.append({'key':key,'source_sha256':digest,'equal_after_dtype_conversion':True})
    return {'metadata_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors.index.json']},'sampled_expert_tensors':rows,'scope':'Two experts per layer, all 48 decoder layers; complete tensors checked. Not a full checkpoint hash.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);evaluation_arguments(p);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    def save(name,obj):(a.out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save('run.json',capture({'backend':'hf','numerics':a.numerics,'dtype':'bfloat16','projection':'selected','max_input_tokens':4000,'seed':37,'device_map':'auto','max_memory_gib_per_gpu':20,'modes':['causal','isolated']},a.model_revision,a.model))
    try:
        torch.manual_seed(37)
        assert torch.cuda.device_count()==4,'Exactly four GPUs required'
        for i in range(4):
            x=torch.randn(8,8,device=f'cuda:{i}');assert torch.isfinite(x@x).all()
            print('WITNESS',i,torch.cuda.get_device_name(i),flush=True)
        groups,excluded=prepare(a.root);selected=[]
        for d in ['clotho_aqa','muchomusic','video_mme_v2','mmou']:
            g=next(g for g in groups if len(g['questions'])>=2 and g['questions'][0]['dataset']==d)
            selected.append({**g,'questions':g['questions'][:2]})
        save('manifest.json',{'groups':selected,'excluded_count':len(excluded),'source_sha256':hashlib.sha256((a.root/'grouped.jsonl').read_bytes()).hexdigest(),'seed':37,'dtype':'bfloat16','projection':'selected','logit_atol':0.1,'probability_atol':0.02,'scope':'4 media, 8 original-label questions; numerical parity and timing only; no aggregate accuracy claim'})
        engine,loading=measure(lambda:HFMJevEngine(a.model,max_memory={i:'20GiB' for i in range(4)},numerics=a.numerics,dtype='bfloat16',max_input_tokens=4000))
        mapping=engine.model.hf_device_map
        assert all(str(v) not in ('cpu','disk') for v in mapping.values()),mapping
        assert set(map(str,mapping.values()))=={'0','1','2','3'},mapping
        engine.model.generate=lambda *x,**k: (_ for _ in ()).throw(AssertionError('generate forbidden'))
        save('checkpoint_witness.json',checkpoint_witness(engine,a.model))
        source_paths=[Path(__file__),Path(__file__).with_name('evaluate_service.py'),Path(sys.modules['mjev.hf'].__file__),Path(sys.modules['mjev.prompt'].__file__)]
        save('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
        for source_path in source_paths:
            dest=a.out/'source'/source_path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(source_path.read_bytes())
        save('environment.json',{'loading':loading,'checkpoint_loading_info':engine.loading_info,'device_map':mapping,'torch':torch.__version__,'gpu_names':[torch.cuda.get_device_name(i) for i in range(4)]})
        print('MODEL_LOADED',loading,flush=True)
        results=[]
        for gi,g in enumerate(selected):
            media=str(a.root/g['media_path']);qs=[{'id':q['id'],'question':q['question'],'candidates':q['candidates']} for q in g['questions']]
            options={'fps':1,'min_pixels':3136,'max_pixels':50176,'max_frames':32} if g['modality']!='audio' else None
            # Warm each media/shape; neither cold decoding nor first CUDA dispatch enters paired timings.
            engine.score_many(media,qs[:1],modality=g['modality'],video_options=options)
            for mode in (['causal','isolated'] if gi%2==0 else ['isolated','causal']):
                timings={};values={};cache=None
                for cached in ([False,True] if gi%2==0 else [True,False]):
                    name='cached' if cached else 'uncached'
                    if cached:
                        cache,pre=measure(lambda:engine.prepare_context(media,modality=g['modality'],video_options=options))
                        cached_values=[];suffix=[]
                        for q in qs:
                            one,t=measure(lambda:engine.score_questions(cache,[q],mode=mode));cached_values.extend(one);suffix.append(t)
                        values[name]=cached_values;timings[name]={'prefill':pre,'questions':suffix,'total_seconds':pre['seconds']+sum(t['seconds'] for t in suffix)}
                    else:
                        values[name],timings[name]=measure(lambda:engine.score_many(media,qs,modality=g['modality'],video_options=options,mode=mode))
                    # Drop KV before measuring an uncached branch so memory accounting is comparable.
                    if cached and gi%2==1:del cache;cache=None;gc.collect();torch.cuda.empty_cache()
                comparisons=[compare(x,y) for x,y in zip(values['uncached'],values['cached'])]
                if cache is None:cache=engine.prepare_context(media,modality=g['modality'],video_options=options)
                reversed_values=engine.score_questions(cache,list(reversed(qs)),mode=mode)
                order_checks=[compare(x,y) for x,y in zip(values['cached'],reversed(reversed_values))]
                # Test a candidate permutation against its own uncached counterpart, not against another labeling.
                perm=dict(qs[0]);perm['candidates']=list(reversed(list(perm['candidates'].values())))
                perm_a=engine.score_many(media,[perm],modality=g['modality'],video_options=options,mode=mode)[0]
                perm_b=engine.score_questions(cache,[perm],mode=mode)[0]
                permutation=compare(perm_a,perm_b)
                row={'dataset':g['questions'][0]['dataset'],'modality':g['modality'],'mode':mode,'timings':timings,'values':values,'comparisons':comparisons,'order_checks':order_checks,'permutation_check':permutation}
                results.append(row);save('results.json',results)
                print('GROUP_DONE',row['dataset'],mode,json.dumps({'timings':timings,'comparisons':comparisons}),flush=True)
                del cache;gc.collect();torch.cuda.empty_cache()
        checks=[c for r in results for c in r['comparisons']+r['order_checks']+[r['permutation_check']]]
        summary={'media':len(selected),'questions':sum(len(g['questions']) for g in selected),'mode_groups':len(results),'max_abs_logit_error':max(c['max_abs_logit_error'] for c in checks),'max_abs_probability_error':max(c['max_abs_probability_error'] for c in checks),'same_answer_checks':sum(c['same_answer'] for c in checks),'total_checks':len(checks),'vllm_imported':any(k=='vllm' or k.startswith('vllm.') for k in sys.modules),'note':'Warmed local in-process timings, sequential questions, includes processor work; loading excluded. Not API latency or throughput/accuracy benchmark.'}
        summary['passed']=all(c['max_abs_logit_error']<=0.1 and c['max_abs_probability_error']<=0.02 and c['same_answer'] for c in checks) and not summary['vllm_imported']
        save('summary.json',summary)
        if not summary['passed']:raise AssertionError('Parity acceptance failed; inspect preserved raw results')
        (a.out/'_SUCCESS').write_text('Full-weight BF16 four-GPU small-sample cache checks passed.\n')
    except Exception as e:
        save('ERROR.json',{'error':repr(e)});raise
if __name__=='__main__':main()
