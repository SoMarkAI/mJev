"""Freeze original annotations and sample questions before downloading media."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse
import requests
from .dataset import validate
from .download import HTTPRangeFile, atomic_json, download, sha256, url, transport_url

SOURCES = {
    'mmau': ('gamma-lab-umd/MMAU-test-mini', 'ccd9696c0111ea7060827598f310558df0b71b0a', 'test_mini.parquet'),
    'video_mme_v2': ('MME-Benchmarks/Video-MME-v2', '6e4bebb03202e1ddbf3d37703e560e51c5aa2d64', 'test.parquet'),
    'mmou': ('nvidia/MMOU', '60fddffb699443e618148f6e3ef84bb63f039cf5', 'MMOU_TEST_MINI.json'),
}
MMOU_MEDIA = ('sonalkum/MMOU-Videos', 'eeb67c9e9bafd187746ac013926133a07cdfac1c')


def labels(n):
    if n > 26: raise ValueError('Source adapter requires explicit labels above Z')
    return [chr(65+i) for i in range(n)]


def pick(items, count, seed, namespace, key):
    if count > len(items): raise ValueError('Sample larger than eligible pool')
    def rank(x):
        return hashlib.sha256(f'{seed}:{namespace}:{key(x)}'.encode()).hexdigest(), str(key(x))
    return sorted(items, key=rank)[:count]


def parse_options(text):
    """Remove only source label delimiters; preserve the option text verbatim."""
    matches = list(re.finditer(r'(?m)^([A-Z])\. ', text))
    if not matches or matches[0].start() != 0: raise ValueError('Unrecognized source options')
    out = {}
    for i,m in enumerate(matches):
        end = matches[i+1].start()-1 if i+1<len(matches) else len(text)
        if m[1] in out: raise ValueError('Duplicate option label')
        out[m[1]] = text[m.end():end]
    return out


def make(dataset, original_id, question, candidates, answer, media_path, task,
         modality, original, **metadata):
    repo, revision, filename = SOURCES[dataset]
    row = {'id': f'{dataset}:{original_id}', 'dataset':dataset,
           'source': {'repository':repo, 'revision':revision, 'file':filename, 'original_id':str(original_id)},
           'split': 'test-mini' if dataset != 'video_mme_v2' else 'test',
           'task': task, 'modality':modality, 'media_path':media_path,
           'media':{'type':'audio' if modality=='audio' else 'video', 'path':media_path},
           'question':question, 'candidates':candidates, 'label':answer,
           'metadata':metadata, 'original':original}
    return validate(row)


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(root, count=1000, seed=20260925):
    import pyarrow.parquet as pq
    root=Path(root);sources=root/'sources';sources.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        old=json.loads((root/'selection.json').read_text())
        if old['seed'] != seed or old['requested_per_dataset'] != count:
            raise ValueError('Existing frozen selection differs: choose a new output directory')
        print('Frozen selection exists; not resampling');return old
    annotations={}
    for key,(repo,rev,name) in SOURCES.items():
        if key=='mmau':
            path=sources/'mmau_metadata.json'
            if not path.exists():
                with HTTPRangeFile(url(repo,rev,name)) as f:
                    p=pq.ParquetFile(f)
                    rows=p.read(columns=['instruction','choices','answer','other_attributes']).to_pylist()
                atomic_json(path,rows)
            annotations[key]=json.loads(path.read_text())
        else:
            path=sources/name
            download(url(repo,rev,name),path)
            annotations[key]=pq.read_table(path).to_pylist() if name.endswith('.parquet') else json.loads(path.read_text())
    # Metadata-only index; no root-level main-set videos or captions downloaded.
    index_file=sources/'mmou_media_index.json'
    if not index_file.exists():
        response=requests.get(transport_url(f'https://huggingface.co/api/datasets/{MMOU_MEDIA[0]}/revision/{MMOU_MEDIA[1]}'),timeout=60)
        response.raise_for_status();atomic_json(index_file,response.json())
    available={x['rfilename'] for x in json.loads(index_file.read_text())['siblings']}
    all_rows=[];plans=[];excluded=[]
    mmau=annotations['mmau']
    if len(mmau)!=1000:raise ValueError('Unexpected MMAU mini size')
    if count!=1000:raise ValueError('This suite keeps all 1000 MMAU-mini rows; other sizes need a new suite design')
    for i,x in enumerate(mmau):
        attrs=json.loads(x['other_attributes']) if isinstance(x['other_attributes'],str) else x['other_attributes']
        matches=[j for j,c in enumerate(x['choices']) if c==x['answer']]
        if len(matches)!=1:raise ValueError('Ambiguous MMAU original answer')
        original_id=attrs.get('id',str(i))
        choice_labels=labels(len(x['choices']))
        choices={}
        for letter,text in zip(choice_labels,x['choices']):
            prefix=f'({letter}) '
            if not text.startswith(prefix):raise ValueError('Unexpected MMAU option prefix')
            choices[letter]=text[len(prefix):]
        all_rows.append(make('mmau',original_id,x['instruction'],choices,labels(len(x['choices']))[matches[0]],
                            f'media/mmau/{i:04d}.audio',attrs.get('task','audio_understanding'),'audio',x,
                            source_row_index=i,attributes=attrs))
    plans.append({'kind':'parquet_audio','dataset':'mmau','url':url(*SOURCES['mmau']),
                  'source_path':'sources/test_mini.parquet','expected_rows':1000})
    groups=defaultdict(list)
    for x in annotations['video_mme_v2']:groups[x['video_id']].append(x)
    if count%4 or any(len(xs)!=4 for xs in groups.values()):raise ValueError('Expected 4-question video groups')
    selected=pick(list(groups),count//4,seed,'video_mme_v2',lambda x:x)
    archives=defaultdict(dict)
    for vid in selected:
        media=f'media/video_mme_v2/{vid}.mp4'
        archives[f'videos/{(int(vid)-1)//20+1:03d}.zip'][f'{vid}.mp4']=media
        for x in groups[vid]:
            all_rows.append(make('video_mme_v2',x['question_id'],x['question'],parse_options(x['options']),x['answer'],
                                media,x['third_head'],'audio_video',x,video_id=vid,group_type=x['group_type'],
                                group_structure=x['group_structure'],level=x['level'],second_head=x['second_head']))
    repo,rev,_=SOURCES['video_mme_v2']
    for archive,targets in sorted(archives.items()):
        plans.append({'kind':'zip_members','dataset':'video_mme_v2','url':url(repo,rev,archive),'targets':targets})
    eligible=[]
    for x in annotations['mmou']:
        vid=parse_qs(urlparse(x['video_url']).query).get('v',[None])[0]
        if not vid or not re.fullmatch(r'[A-Za-z0-9_-]{11}',vid):raise ValueError('Invalid source video ID')
        source_media='test/'+vid+'.mp4'
        if source_media not in available:
            excluded.append({'id':x['question_id'],'reason':'video absent from pinned test media index','video_file':source_media})
        else:eligible.append((x,vid,source_media))
    selected=pick(eligible,count,seed,'mmou',lambda x:x[0]['question_id'])
    videos={}
    for x,vid,source_media in selected:
        media=f'media/mmou/{vid}.mp4';videos[source_media]=media
        all_rows.append(make('mmou',x['question_id'],x['question'],dict(x['options']),x['correct_option_letter'],
                            media,x['question_type'],'audio_video',x,video_id=vid,
                            video_duration=x['video_duration'],evidence_start=x['start_time'],evidence_end=x['end_time'],
                            media_repository=MMOU_MEDIA[0],media_revision=MMOU_MEDIA[1]))
    for name,target in sorted(videos.items()):
        plans.append({'kind':'file','dataset':'mmou','url':url(*MMOU_MEDIA,name),'target':target})
    ids=[r['id'] for r in all_rows]
    if len(set(ids))!=len(ids):raise ValueError('Nonunique source IDs')
    write_jsonl(root/'manifest.jsonl',all_rows)
    for key in SOURCES:write_jsonl(root/(key+'.jsonl'),[r for r in all_rows if r['dataset']==key])
    atomic_json(root/'download_plan.json',plans)
    report={'schema_version':'mjev-benchmark-v1','seed':seed,'requested_per_dataset':count,
            'sampling':{'mmau':'all original 1000 rows','video_mme_v2':'SHA256-ranked 250 complete video groups, 4 questions each',
                        'mmou':'SHA256-ranked questions among entries with video listed in pinned test media index'},
            'counts':dict(Counter(r['dataset'] for r in all_rows)),
            'unique_media':{k:len({r['media_path'] for r in all_rows if r['dataset']==k}) for k in SOURCES},
            'candidate_counts':{k:dict(Counter(len(r['candidates']) for r in all_rows if r['dataset']==k)) for k in SOURCES},
            'mmou_pool_size':len(annotations['mmou']),'mmou_eligible':len(eligible),'mmou_excluded':excluded,
            'manifest_sha256':sha256(root/'manifest.jsonl'),
            'source_annotation_sha256':{p.name:sha256(p) for p in sources.iterdir() if p.is_file() and not p.name.endswith('.receipt.json')},
            'source_revisions':SOURCES,'mmou_media_revision':MMOU_MEDIA}
    atomic_json(root/'selection.json',report);print(json.dumps({k:report[k] for k in ['counts','unique_media','candidate_counts','mmou_eligible']},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20260925)
    args=p.parse_args();prepare(args.output,seed=args.seed)

if __name__=='__main__':main()
