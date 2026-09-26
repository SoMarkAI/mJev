"""Verify annotation fidelity, frozen selection and every downloaded media file."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
from .dataset import MJevDataset
from .download import atomic_json, sha256


def annotation_fidelity(row):
    source=row['original'];dataset=row['dataset']
    if dataset=='mmau':
        assert row['question']==source['instruction']
        assert [f'({k}) {v}' for k,v in row['candidates'].items()]==source['choices']
        assert f"({row['label']}) {row['candidates'][row['label']]}"==source['answer']
    elif dataset=='video_mme_v2':
        assert row['question']==source['question']
        assert '\n'.join(f'{k}. {v}' for k,v in row['candidates'].items())==source['options']
        assert row['label']==source['answer']
    elif dataset=='mmou':
        assert row['question']==source['question']
        assert list(row['candidates'].items())==list(source['options'].items())
        assert row['label']==source['correct_option_letter']
    else:raise ValueError('Unknown dataset')


def duration_warnings(records, media_index):
    measured={x['media_path']:x['duration_seconds'] for x in media_index}
    warnings={}
    for row in records:
        declared=row.get('metadata',{}).get('video_duration')
        if declared is None or row['media_path'] not in measured:continue
        actual=measured[row['media_path']]
        if abs(actual-float(declared))>2.0:
            item=warnings.setdefault(row['media_path'],{'media_path':row['media_path'],
                'annotation_duration_seconds':declared,'file_duration_seconds':actual,
                'difference_seconds':actual-float(declared),'question_ids':[]})
            item['question_ids'].append(row['id'])
    return list(warnings.values())


def audit(root, ffprobe='ffprobe'):
    root=Path(root)
    (root/'_SUCCESS').unlink(missing_ok=True)
    selection=json.loads((root/'selection.json').read_text())
    if sha256(root/'manifest.jsonl')!=selection['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    ds=MJevDataset(root/'manifest.jsonl',require_media=False)
    import pyarrow.parquet as pq
    originals={
        'mmau':json.loads((root/'sources/mmau_metadata.json').read_text()),
        'video_mme_v2':{x['question_id']:x for x in pq.read_table(root/'sources/test.parquet').to_pylist()},
        'mmou':{x['question_id']:x for x in json.loads((root/'sources/MMOU_TEST_MINI.json').read_text())}}
    for name,digest in selection['source_annotation_sha256'].items():
        if sha256(root/'sources'/name)!=digest:raise ValueError('Source snapshot changed: '+name)
    audio_blob_path=root/'sources/mmau_media_blobs.json'
    if audio_blob_path.exists():
        upstream=next(x for x in json.loads(audio_blob_path.read_text())['siblings'] if x['rfilename']=='test_mini.parquet')
        if sha256(root/'sources/test_mini.parquet')!=upstream['lfs']['sha256']:
            raise ValueError('MMAU source Parquet differs from pinned upstream LFS object')
    for r in ds:
        annotation_fidelity(r)
        source_index=r['metadata']['source_row_index'] if r['dataset']=='mmau' else r['source']['original_id']
        if r['original']!=originals[r['dataset']][source_index]:raise ValueError('Original differs from source annotation file')
    if dict(Counter(r['dataset'] for r in ds))!=selection['counts']:raise ValueError('Selection count mismatch')
    for dataset in selection['counts']:
        subset=MJevDataset(root/(dataset+'.jsonl'),require_media=False)
        if subset.records!=[r for r in ds if r['dataset']==dataset]:raise ValueError('Per-dataset manifest differs')
    paths={r['media_path']:r for r in ds};verified=[];errors=[]
    blobs_path=root/'sources/mmou_media_blobs.json'
    blobs={x['rfilename']:x for x in json.loads(blobs_path.read_text())['siblings']} if blobs_path.exists() else {}
    for relative,row in sorted(paths.items()):
        path=root/relative
        try:
            receipt=json.loads(path.with_suffix(path.suffix+'.receipt.json').read_text())
            if path.stat().st_size!=receipt['bytes'] or sha256(path)!=receipt['sha256']:
                raise ValueError('Media does not match download receipt')
            if row['dataset']=='mmou' and blobs:
                upstream=blobs['test/'+Path(relative).name]
                if upstream['lfs']['sha256']!=receipt['sha256'] or upstream['size']!=receipt['bytes']:
                    raise ValueError('Mismatch against pinned upstream LFS metadata')
            cmd=[ffprobe,'-v','error','-show_entries','format=duration:stream=codec_type,codec_name,width,height,sample_rate,channels',
                 '-of','json',str(path)]
            result=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=60)
            probe=json.loads(result.stdout)
            kinds=[x['codec_type'] for x in probe.get('streams',[])]
            if row['modality']=='audio' and 'audio' not in kinds:raise ValueError('No audio stream')
            if row['modality'] in ('video','audio_video') and 'video' not in kinds:raise ValueError('No video stream')
            duration=float(probe.get('format',{}).get('duration',0))
            if duration<=0:raise ValueError('Invalid media duration')
            verified.append({'media_path':relative,'bytes':receipt['bytes'],'sha256':receipt['sha256'],
                             'duration_seconds':duration,'streams':probe['streams'],
                             'has_audio':'audio' in kinds})
        except Exception as exc:
            errors.append({'media_path':relative,'error':type(exc).__name__,'detail':str(exc)[:300]})
    expected=set(paths)
    unexpected=[str(p.relative_to(root)) for p in (root/'media').rglob('*')
                if p.is_file() and p.suffix in ('.audio','.wav','.mp4') and str(p.relative_to(root)) not in expected]
    report={'verified_at':time.time(),'question_count':len(ds),'original_annotations_preserved':True,
            'counts':dict(Counter(r['dataset'] for r in ds)), 'unique_media_expected':len(paths),
            'unique_media_verified':len(verified),'media_bytes':sum(x['bytes'] for x in verified),
            'videos_without_audio':[x['media_path'] for x in verified if x['media_path'].endswith('.mp4') and not x['has_audio']],
            'errors':errors,'unexpected_media':unexpected,'manifest_sha256':selection['manifest_sha256'],
            'validation':'file SHA256 plus ffprobe container/stream metadata; not a full decode of every frame'}
    warnings=duration_warnings(ds,verified)
    report['source_duration_warning_videos']=len(warnings)
    atomic_json(root/'duration_warnings.json',warnings)
    atomic_json(root/'media_index.json',verified);atomic_json(root/'validation.json',report)
    atomic_json(root/'status.json', {'state':'incomplete' if errors or unexpected else 'complete',
                                   'questions':len(ds),'media_expected':len(paths),'media_verified':len(verified),
                                   'errors':len(errors),'model_inference_performed':False})
    success=root/'_SUCCESS'
    if errors or unexpected:
        success.unlink(missing_ok=True)
        raise RuntimeError(f'{len(errors)} missing/invalid media files, {len(unexpected)} unexpected files')
    success.write_text('All selected original questions and selected media verified. No model evaluation performed.\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--ffprobe',default='ffprobe')
    args=p.parse_args();print(json.dumps(audit(args.root,args.ffprobe),indent=2))

if __name__=='__main__':main()
