"""Model-independent grouped evaluation with explicit settings and immutable provenance."""
import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from .dataset import MJevDataset
from .evaluate import evaluate
from .provenance import capture, revision, sha256, write
from .prepare import write_jsonl
from .artifacts import atomic_json, atomic_jsonl

REQUIRED={'model_id','model_revision','backend','numerics','mode','dtype','projection',
          'prefix_cache','question_batch_size','max_input_tokens','context','video_options',
          'seed','tensor_parallel_size'}


def config_load(path):
    c=json.loads(Path(path).read_text())
    if set(c)!=REQUIRED:raise ValueError('Every inference setting must be explicit; unknown settings rejected')
    revision(c['model_revision'])
    if c['model_id'] not in ('Qwen/Qwen3-VL-4B-Instruct','Qwen/Qwen3-Omni-30B-A3B-Instruct'):raise ValueError('Unsupported official model')
    if c['backend'] not in ('hf','vllm') or c['mode'] not in ('causal','isolated') or c['numerics'] not in ('stable','native'):raise ValueError('Invalid inference mode')
    if c['dtype']!='bfloat16' or c['projection'] not in ('full','selected'):raise ValueError('Invalid dtype/projection')
    for k in ['question_batch_size','max_input_tokens','tensor_parallel_size']:
        if type(c[k]) is not int or c[k]<1:raise ValueError(f'Invalid {k}')
    if type(c['prefix_cache']) is not bool:raise ValueError('Invalid prefix_cache')
    if c['backend']=='vllm' and c['projection']!='full':raise ValueError('vLLM readout uses full projection')
    return c


async def run(manifest, config_path, out):
    c=config_load(config_path);data=MJevDataset(manifest)
    from mjev.families import check_modality, VL, OMNI
    kind=VL if c['model_id']=='Qwen/Qwen3-VL-4B-Instruct' else OMNI
    for row in data:check_modality(kind,row['modality'])
    if not len(data):raise ValueError('Empty dataset')
    if os.environ.get('MJEV_ENABLE')=='1' or os.environ.get('MJEV_ENABLE_PATCHES')=='1':raise ValueError('Start with runtime hooks disabled')
    os.environ['VLLM_BATCH_INVARIANT']='1' if c['numerics']=='stable' else '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    out=Path(out);out.mkdir(parents=True,exist_ok=False);engine=None
    try:
        from huggingface_hub import snapshot_download
        model=snapshot_download(c['model_id'],revision=c['model_revision'])
        import torch
        torch.manual_seed(c['seed'])
        def sync():
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
        evidence=capture(c,c['model_revision'],model)
        evidence['model_revision_status']='downloaded_or_cached_hub_snapshot_at_requested_commit'
        evidence['manifest_sha256']=sha256(manifest)
        evidence['media_sha256']={r['media_path']:sha256(data.media_path(r)) for r in data}
        evidence['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        write(out/'run.json',evidence);write_jsonl(out/'manifest.jsonl',data.records)
        start=time.perf_counter()
        if c['backend']=='hf':
            from mjev.hf import HFMJevEngine
            engine=HFMJevEngine(model,numerics=c['numerics'],dtype=c['dtype'],max_input_tokens=c['max_input_tokens'])
        else:
            os.environ['MJEV_ENABLE']='1'
            from mjev.patch import install
            install()
            from mjev.engine import MJevEngine
            engine=MJevEngine(model,enable_av=True,enable_prefix_caching=c['prefix_cache'],
                max_model_len=c['max_input_tokens'],tensor_parallel_size=c['tensor_parallel_size'])
        sync();startup=time.perf_counter()-start
        groups=defaultdict(list)
        for row in data:groups[(row['modality'],row['media_path'])].append(row)
        predictions=[];raw=[]
        (out/'groups').mkdir()
        atomic_json(out/'progress.json', {'completed_groups':0,'total_groups':len(groups),'completed_questions':0})
        for group_index,((modality,media),rows) in enumerate(groups.items()):
            qs=[{'id':r['id'],'question':r['question'],'candidates':list(r['candidates'].values())} for r in rows]
            from mjev.prompt import PromptBuilder
            if any(list(r['candidates'])!=PromptBuilder.labels(len(r['candidates'])) for r in rows):raise ValueError('Labels must preserve ordered A/B/... mapping')
            sync();start=time.perf_counter()
            if c['backend']=='hf':
                results=engine.score_many(str(data.media_path(rows[0])),qs,modality=modality,
                    context=c['context'],mode=c['mode'],projection=c['projection'],
                    video_options=c['video_options'] if modality in ('video','audio_video') else None,
                    use_prefix_cache=c['prefix_cache'],batch_size=c['question_batch_size'])
            else:
                results=[]
                for offset in range(0,len(qs),c['question_batch_size']):
                    results.extend(await asyncio.gather(*(engine.score_media(str(data.media_path(rows[0])),q['question'],q['candidates'],modality=modality,
                        context=c['context'],mode=c['mode'],video_options=c['video_options'] if modality in ('video','audio_video') else None)
                        for q in qs[offset:offset+c['question_batch_size']])))
            sync();elapsed=time.perf_counter()-start
            if len(results)!=len(rows):raise ValueError('Incomplete or extra scorer output')
            for row,result in zip(rows,results):
                if result.get('id') is not None and result['id'] != row['id']:raise ValueError('Question identity changed')
                if [x['text'] for x in result['candidates']]!=list(row['candidates'].values()):raise ValueError('Candidate text/order changed')
                predictions.append({'id':row['id'],'logits':{x['label']:x['raw_logit'] for x in result['candidates']}})
            raw.append({'ids':[r['id'] for r in rows],'modality':modality,'group_seconds':elapsed,'results':results})
            # A group file is the durable unit of completion. A later failure
            # retains all previously scored groups; this does not imply resume support.
            atomic_json(out/'groups'/f'{group_index:06d}.json', {
                'status':'complete', 'group_index':group_index, 'media_path':media,
                **raw[-1], 'predictions':predictions[-len(rows):]})
            atomic_jsonl(out/'predictions.jsonl', predictions)
            atomic_jsonl(out/'raw.jsonl', raw)
            atomic_json(out/'progress.json', {'completed_groups':group_index+1,
                'total_groups':len(groups), 'completed_questions':len(predictions)})
        result={'scope':'Evaluation of the supplied frozen manifest; synthetic fixtures are smoke tests, not benchmark accuracy',
            'evaluation':evaluate(data.records,predictions,bins=15),'run':evidence,
            'timing':{'startup_seconds':startup,'groups':[{'ids':r['ids'],'seconds':r['group_seconds']} for r in raw],
                'boundary':'Includes decoding, processing and scoring; excludes model load; no warmup; not steady-state API latency'},
            'predictions_sha256':sha256(out/'predictions.jsonl'),'raw_sha256':sha256(out/'raw.jsonl')}
        write(out/'report.json',result)
        (out/'REPORT.md').write_text('# Grouped evaluation\n\n'+result['scope']+'\n\n'+json.dumps(result['evaluation']['metrics'],indent=2)+'\n')
        (out/'_SUCCESS').write_text('All manifest questions evaluated.\n')
    except Exception as exc:
        write(out/'ERROR.json',{'error':str(exc)});raise
    finally:
        if c['backend']=='vllm' and engine is not None:engine.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();asyncio.run(run(a.manifest,a.config,a.out))

if __name__=='__main__':main()
