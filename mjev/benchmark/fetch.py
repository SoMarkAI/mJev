"""Download only frozen media plan; resumable, bounded concurrency, fail-stop."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
import pyarrow.parquet as pq
from .download import atomic_json, download, extract_selected_zip, sha256


def run_job(root, job):
    if job['kind']=='file':
        return [download(job['url'], root/job['target'])]
    if job['kind']=='zip_members':
        return extract_selected_zip(job['url'], {k:root/v for k,v in job['targets'].items()})
    if job['kind']=='parquet_audio':
        path=root/job['source_path']
        source_receipt=download(job['url'],path)
        meta=json.loads((root/'sources/mmau_metadata.json').read_text())
        p=pq.ParquetFile(path)
        if p.metadata.num_rows!=job['expected_rows']:raise ValueError('Wrong audio row count')
        index=0;receipts=[]
        for batch in p.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if {k:v for k,v in row.items() if k!='context'}!=meta[index]:
                    raise ValueError('Audio and frozen annotations differ')
                data=row['context']['bytes']
                if not isinstance(data,bytes) or not data:raise ValueError('Missing embedded audio')
                target=root/f'media/mmau/{index:04d}.audio';target.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256(data).hexdigest()
                if not target.exists() or sha256(target)!=digest:
                    part=target.with_suffix('.audio.part');part.write_bytes(data);part.replace(target)
                record={'url':job['url'],'row_index':index,'original_audio_path':row['context']['path'],
                        'bytes':len(data),'sha256':digest,'parquet_sha256':source_receipt['sha256']}
                atomic_json(target.with_suffix('.audio.receipt.json'),record)
                receipts.append(record);index+=1
        if index!=job['expected_rows']:raise ValueError('Incomplete audio extraction')
        return receipts
    raise ValueError('Unknown download job')


def fetch(root,workers=6,limit=None,dataset=None):
    root=Path(root);jobs=json.loads((root/'download_plan.json').read_text())
    if dataset:jobs=[j for j in jobs if j['dataset']==dataset]
    if limit:jobs=jobs[:limit]
    state={'state':'running','started_at':time.time(),'jobs_total':len(jobs),
           'jobs_completed':0,'media_completed':0,'media_bytes':0,'errors':[],
           'dataset_filter':dataset,'limit':limit}
    status=root/('download_status'+('_'+dataset if dataset else '')+'.json')
    atomic_json(status,state)
    iterator=iter(enumerate(jobs));pending={}
    def submit(executor):
        try:i,j=next(iterator)
        except StopIteration:return False
        pending[executor.submit(run_job,root,j)]=(i,j);return True
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers):
            if not submit(executor):break
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                i,job=pending.pop(future)
                try:
                    records=future.result()
                    state['jobs_completed']+=1;state['media_completed']+=len(records)
                    state['media_bytes']+=sum(r['bytes'] for r in records)
                    print(json.dumps({'job':i,'dataset':job['dataset'],'media':len(records),
                                      'jobs_completed':state['jobs_completed'],'jobs_total':len(jobs)}),flush=True)
                except Exception as exc:
                    # Avoid recording signed redirect URLs or credentials from exception text.
                    state['errors'].append({'job':i,'dataset':job['dataset'],'type':type(exc).__name__,
                                            'source_url':job['url'],'message':str(exc).split('?')[0][:300]})
                    state['state']='pausing_on_error'
                state['updated_at']=time.time();atomic_json(status,state)
            if not state['errors']:
                for _ in done:submit(executor)
    state['state']='paused_on_error' if state['errors'] else 'download_complete'
    state['finished_at']=time.time();atomic_json(status,state)
    if state['errors']:raise RuntimeError('Download paused; see status. No automatic retries were issued.')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int)
    p.add_argument('--dataset',choices=['mmau','mmou','video_mme_v2'])
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    print(json.dumps(fetch(args.root,args.workers,args.limit,args.dataset),indent=2))

if __name__=='__main__':main()
