"""Public, revision-pinned MMOU mini workflow. No internal bundles required."""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlparse

from .dataset import MJevDataset
from .download import download, url
from .evaluate import evaluate
from .prepare import SOURCES, MMOU_MEDIA, make, pick, write_jsonl
from .provenance import capture, revision, sha256, write


def duration_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(':')
    total = 0.
    for part in parts:
        total = total * 60 + float(part)
    return total


def select_groups(annotations, media_names, videos, max_seconds, seed):
    groups = defaultdict(list)
    excluded = []
    for item in annotations:
        vid = parse_qs(urlparse(item['video_url']).query).get('v', [None])[0]
        if not vid or not re.fullmatch(r'[A-Za-z0-9_-]{11}', vid):
            raise ValueError('Invalid source video ID')
        groups[vid].append(item)
    eligible = []
    for vid, questions in sorted(groups.items()):
        if 'test/'+vid+'.mp4' not in media_names:
            excluded.append({'video_id': vid, 'reason': 'missing_from_pinned_index'})
        elif any(not 0 < duration_seconds(q['video_duration']) <= max_seconds for q in questions):
            excluded.append({'video_id': vid, 'reason': 'annotation_duration_filter'})
        else:
            eligible.append((vid, sorted(questions, key=lambda q: str(q['question_id']))))
    selected = pick(eligible, videos, seed, 'public-mmou-video-v1', lambda g: g[0])
    return selected, excluded


def prepare(root, videos=2, max_seconds=15., seed=20260926):
    if videos < 1 or max_seconds <= 0:
        raise ValueError('videos and max-seconds must be positive')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    try:
        annotation = root/'sources/annotations.json'
        download(url(*SOURCES['mmou']), annotation)
        index = root/'sources/media-index.json'
        download(f'https://huggingface.co/api/datasets/{MMOU_MEDIA[0]}/revision/{MMOU_MEDIA[1]}', index)
        names = {x['rfilename'] for x in json.loads(index.read_text())['siblings']}
        selected, excluded = select_groups(json.loads(annotation.read_text()), names, videos, max_seconds, seed)
        rows, plans = [], []
        for vid, questions in selected:
            relative = f'media/{vid}.mp4'
            plans.append({'url': url(*MMOU_MEDIA, f'test/{vid}.mp4'), 'target': relative})
            for q in questions:
                row = make('mmou', q['question_id'], q['question'], dict(q['options']), q['correct_option_letter'],
                           relative, q['question_type'], 'audio_video', q, video_id=vid,
                           video_duration=q['video_duration'], evidence_start=q['start_time'], evidence_end=q['end_time'])
                row.update(source_dataset='MMOU Test Mini', source_id=str(q['question_id']),
                           source_url='https://huggingface.co/datasets/nvidia/MMOU', license='Apache-2.0',
                           media_source_url=url(*MMOU_MEDIA, f'test/{vid}.mp4'),
                           license_details={'annotation_declaration':'https://huggingface.co/datasets/nvidia/MMOU/blob/'+SOURCES['mmou'][1]+'/README.md',
                                            'media_repository_declared_license':'Apache-2.0',
                                            'limitation':'Underlying video/audio rights unverified; see THIRD_PARTY_LICENSES.md'},
                           media_license='UNKNOWN', redistribution_allowed='unknown')
                rows.append(row)
        write_jsonl(root/'manifest.jsonl', rows)
        write(root/'selection.json', {'schema':'public-mmou-mini-v1','seed':seed,'videos':videos,
              'max_annotation_seconds':max_seconds,'selection':'SHA256-ranked eligible video groups; all original questions',
              'source_revision':SOURCES['mmou'],'media_revision':MMOU_MEDIA,'excluded':excluded,
              'annotation_sha256':sha256(annotation),'manifest_sha256':sha256(root/'manifest.jsonl'),
              'questions':len(rows),'download_plan':plans})
        receipts = {}
        for plan in plans:
            receipts[plan['target']] = download(plan['url'], root/plan['target'])
        write(root/'media_receipts.json', receipts)
        verify(root)
        (root/'_DATA_READY').write_text('Source labels retained; selected media bytes verified. Not full decode or inference.\n')
    except Exception as exc:
        write(root/'ERROR.json', {'stage':'prepare','error':str(exc)})
        raise


def verify(root):
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text())
    if sha256(root/'manifest.jsonl') != selection['manifest_sha256']:
        raise ValueError('Manifest changed')
    receipts=json.loads((root/'media_receipts.json').read_text())
    dataset=MJevDataset(root/'manifest.jsonl')
    expected={r['media_path'] for r in dataset}
    if set(receipts)!=expected:
        raise ValueError('Media receipt coverage mismatch')
    for name,receipt in receipts.items():
        p=root/name
        if p.stat().st_size!=receipt['bytes'] or sha256(p)!=receipt['sha256']:
            raise ValueError('Media changed: '+name)
    return dataset


_REQUIRED={'backend','numerics','mode','dtype','projection','prefix_cache','question_batch_size',
           'max_input_tokens','device_map','context','video_options','seed','model_id','model_revision'}

def config_load(path):
    config=json.loads(Path(path).read_text())
    if set(config)!=_REQUIRED:
        raise ValueError(f'Config keys must be explicit: missing={_REQUIRED-set(config)}, unknown={set(config)-_REQUIRED}')
    revision(config['model_revision'])
    if config['model_id']!='Qwen/Qwen3-Omni-30B-A3B-Instruct':
        raise ValueError('This workflow targets the official Qwen3-Omni model')
    if config['backend'] not in ('hf','vllm') or config['numerics'] not in ('stable','native') or config['mode'] not in ('causal','isolated'):
        raise ValueError('Invalid backend/numerics/mode')
    if config['dtype']!='bfloat16' or config['projection'] not in ('full','selected'):
        raise ValueError('Use BF16 weights and a supported projection')
    if type(config['prefix_cache']) is not bool or type(config['question_batch_size']) is not int or config['question_batch_size']<1:
        raise ValueError('Invalid cache or batch setting')
    if config['max_input_tokens']!=4000 or config['device_map']!='auto':
        raise ValueError('This mini profile uses full-input limit 4000 and HF auto placement')
    if config['video_options']!={'fps':1,'min_pixels':3136,'max_pixels':50176,'max_frames':8}:
        raise ValueError('Use the explicit published mini video policy')
    if config['backend']=='vllm' and (config['projection']!='full' or config['question_batch_size']!=1 or config['prefix_cache']):
        raise ValueError('Public vLLM mini is serial/full projection/no prefix cache; use separate cache probes')
    return config


async def infer(root, config_path, out):
    import os
    config=config_load(config_path)
    dataset=verify(root)
    if not (Path(root)/'_DATA_READY').exists() or (Path(root)/'ERROR.json').exists():
        raise ValueError('Data preparation incomplete')
    out=Path(out);out.mkdir(parents=True, exist_ok=False)
    engine=None
    try:
        # Set before importing either backend. No global hooks may already be enabled.
        if os.environ.get('MJEV_ENABLE')=='1' or os.environ.get('MJEV_ENABLE_PATCHES')=='1':
            raise ValueError('Start this runner with both mJev hook flags disabled')
        os.environ['VLLM_BATCH_INVARIANT']='1' if config['numerics']=='stable' else '0'
        os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
        pending=capture(config,config['model_revision'])
        pending['model_revision_status']='requested_not_resolved'
        write(out/'run.json',pending)
        from huggingface_hub import snapshot_download
        model=snapshot_download(config['model_id'], revision=config['model_revision'])
        evidence=capture(config, config['model_revision'], model)
        evidence['model_revision_status']='downloaded_or_cached_hub_snapshot_at_requested_commit'
        evidence['weight_integrity']='Hub snapshot provenance; weight shards not independently rehashed'
        evidence['engine_settings']={'vllm_tp':4,'max_num_seqs':8,'max_num_batched_tokens':8192,'gpu_memory_utilization':0.88,'enforce_eager':True,'enable_chunked_prefill':False,'async_scheduling':False,'mm_processor_cache_gb':0,'max_model_len':4000} if config['backend']=='vllm' else {'placement':'device_map=auto','stable_auto_headroom_gib_per_gpu':4,'attention':'sdpa'}
        evidence['manifest_sha256']=sha256(dataset.manifest)
        evidence['media_sha256']={name:r['sha256'] for name,r in json.loads((Path(root)/'media_receipts.json').read_text()).items()}
        write(out/'run.json',evidence)
        write_jsonl(out/'manifest.jsonl',dataset.records)
        import torch
        torch.manual_seed(config['seed'])
        evidence['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        write(out/'run.json',evidence)
        def sync():
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
        start=time.perf_counter()
        if config['backend']=='hf':
            from mjev.hf import HFMJevEngine
            engine=HFMJevEngine(model,numerics=config['numerics'],dtype=config['dtype'],device_map=config['device_map'],max_input_tokens=config['max_input_tokens'])
        else:
            os.environ['MJEV_ENABLE']='1'
            from mjev.patch import install
            install()
            from mjev.engine import MJevEngine
            engine=MJevEngine(model,enable_av=True,enable_prefix_caching=False,max_model_len=4000,mm_processor_cache_gb=0)
        sync();startup=time.perf_counter()-start
        groups=defaultdict(list)
        for row in dataset:groups[row['media_path']].append(row)
        for media,rows in groups.items():
            # Ground truth, original annotations and evidence timestamps never reach the scorer.
            questions=[{'id':r['id'],'question':r['question'],'candidates':list(r['candidates'].values())} for r in rows]
            for row in rows:
                if list(row['candidates'])!=[chr(65+i) for i in range(len(row['candidates']))]:
                    raise ValueError('Original labels do not match the model label mapping')
            sync();start=time.perf_counter()
            if config['backend']=='hf':
                results=engine.score_many(str(dataset.media_path(rows[0])),questions,modality='audio_video',
                    context=config['context'],mode=config['mode'],projection=config['projection'],
                    video_options=config['video_options'],use_prefix_cache=config['prefix_cache'],batch_size=config['question_batch_size'])
            else:
                results=[]
                for q in questions:
                    results.append(await engine.score_media(str(dataset.media_path(rows[0])),q['question'],q['candidates'],
                        modality='audio_video',context=config['context'],mode=config['mode'],video_options=config['video_options']))
            sync();elapsed=time.perf_counter()-start
            if len(results)!=len(rows):raise ValueError('Incomplete scorer output')
            with (out/'predictions.jsonl').open('a') as f:
                for row,result in zip(rows,results):
                    if [c['text'] for c in result['candidates']]!=list(row['candidates'].values()):raise ValueError('Scorer reordered choices')
                    f.write(json.dumps({'id':row['id'],'logits':{c['label']:c['raw_logit'] for c in result['candidates']}},allow_nan=False)+'\n')
            with (out/'raw.jsonl').open('a') as f:
                f.write(json.dumps({'ids':[r['id'] for r in rows],'group_seconds':elapsed,'results':results},allow_nan=False)+'\n')
        write(out/'timing.json',{'startup_seconds':startup,'boundary':'group wall time includes decoding, processor and scoring; excludes model load; no warmup; not API latency or steady-state throughput'})
        evidence['environment']['MJEV_ENABLE']=os.environ.get('MJEV_ENABLE')
        write(out/'run.json',evidence)
        (out/'_INFERENCE_COMPLETE').write_text('Every selected question scored; report pending.\n')
    except Exception as exc:
        write(out/'ERROR.json',{'stage':'infer','error':str(exc)})
        raise
    finally:
        if config['backend']=='vllm' and engine is not None:engine.close()


def report(out):
    out=Path(out)
    (out/'_SUCCESS').unlink(missing_ok=True)
    if not (out/'_INFERENCE_COMPLETE').exists() or (out/'ERROR.json').exists():raise ValueError('Inference incomplete or failed')
    rows=[json.loads(s) for s in (out/'manifest.jsonl').read_text().splitlines()]
    predictions=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
    run=json.loads((out/'run.json').read_text())
    if sha256(out/'manifest.jsonl')!=run['manifest_sha256']:raise ValueError('Manifest differs from recorded run')
    result={'evaluation':evaluate(rows,predictions,bins=15),'metric_config':{'ece_bins':15,'allow_partial':False},'run':run,'timing':json.loads((out/'timing.json').read_text()),
            'predictions_sha256':sha256(out/'predictions.jsonl'),
            'raw_sha256':sha256(out/'raw.jsonl'),'timing_sha256':sha256(out/'timing.json'),
            'scope':'Original-label short-video MMOU subset; not official leaderboard or historical paired reproduction'}
    write(out/'report.json',result)
    m=result['evaluation']['metrics']
    (out/'REPORT.md').write_text('# Public MMOU mini evaluation\n\n'+result['scope']+'\n\n'+
        '\n'.join(f'- {k}: {m[k]}' for k in ('count','accuracy','nll','brier_score','ece'))+
        '\n\nSee report.json for exact configuration, provenance and timing boundaries.\n')
    (out/'_SUCCESS').write_text('Complete predictions evaluated against all frozen labels.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',type=Path,required=True);p.add_argument('--videos',type=int,default=2);p.add_argument('--max-seconds',type=float,default=15);p.add_argument('--seed',type=int,default=20260926)
    p=sub.add_parser('infer');p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('report');p.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.root,a.videos,a.max_seconds,a.seed)
    elif a.command=='infer':asyncio.run(infer(a.root,a.config,a.out))
    else:report(a.out)

if __name__=='__main__':main()
