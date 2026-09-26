"""Run a user-supplied candidate scorer without forwarding reference answers."""
import argparse
import asyncio
import importlib
import inspect
import hashlib
import json
from pathlib import Path
from .dataset import MJevDataset
from .evaluate import score_row


async def run(dataset, scorer, output, concurrency=3, run_config=None):
    if concurrency<1:raise ValueError('concurrency must be positive')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    provenance={'manifest_sha256':hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                'scorer':scorer.__module__+':'+scorer.__qualname__, 'config':run_config or {}}
    sidecar=output.with_suffix(output.suffix+'.run.json')
    if sidecar.exists():
        if json.loads(sidecar.read_text())!=provenance:
            raise ValueError('Run configuration or manifest changed; use a new prediction output')
    elif output.exists() and output.stat().st_size:
        raise ValueError('Existing predictions lack run provenance; use a new output')
    else:
        sidecar.write_text(json.dumps(provenance,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    done={}
    if output.exists():
        for line in output.read_text().splitlines():
            p=json.loads(line)
            if p['id'] in done:raise ValueError('Duplicate existing predictions')
            done[p['id']]=p
    ids={r['id'] for r in dataset}
    if set(done)-ids:raise ValueError('Existing output belongs to a different dataset')
    for row in dataset:
        if row['id'] in done:score_row(row,done[row['id']])
    lock=asyncio.Lock();semaphore=asyncio.Semaphore(concurrency);failed=asyncio.Event()
    async def one(row):
        async with semaphore:
            if failed.is_set() or row['id'] in done:return
            try:
                payload=dataset.model_input(row)
                if inspect.iscoroutinefunction(scorer):result=await scorer(payload)
                else:result=await asyncio.to_thread(scorer,payload)
                if inspect.isawaitable(result):result=await result
                if 'id' in result and result['id']!=row['id']:raise ValueError('Scorer returned wrong ID')
                prediction={'id':row['id'],**result}
                score_row(row,prediction)
                async with lock:
                    with output.open('a') as f:
                        f.write(json.dumps(prediction,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            except Exception:
                failed.set();raise
    # In-flight requests finish; no new calls or retries after the first failure.
    results=await asyncio.gather(*(one(row) for row in dataset),return_exceptions=True)
    failures=[x for x in results if isinstance(x,BaseException)]
    if failures:raise RuntimeError(f'{len(failures)} scorer call(s) failed; output is resumable, no retries issued') from failures[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--scorer',required=True,help='module:function accepting ground-truth-free payload')
    p.add_argument('--output',required=True);p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--run-config',required=True,help='JSON containing model revision, preprocessing and scorer settings')
    args=p.parse_args();module,name=args.scorer.split(':',1)
    scorer=getattr(importlib.import_module(module),name)
    asyncio.run(run(MJevDataset(args.manifest),scorer,args.output,args.concurrency,json.loads(Path(args.run_config).read_text())))

if __name__=='__main__':main()
