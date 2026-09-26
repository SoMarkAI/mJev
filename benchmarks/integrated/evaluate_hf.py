"""Evaluate HF scoring with the same original-label MCQ coverage as vLLM."""
import argparse,json,time,statistics
from pathlib import Path
from mjev.benchmark.provenance import evaluation_arguments, capture, write
from evaluate_service import prepare
from mjev.hf import HFMJevEngine
from mjev.benchmark.evaluate import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['causal','isolated'],default='causal');p.add_argument('--limit-groups',type=int);p.add_argument('--device-map',default='auto');p.add_argument('--dtype',default='bfloat16');p.add_argument('--prefix-cache',action='store_true');p.add_argument('--projection',choices=['full','selected'],required=True);p.add_argument('--question-batch-size',type=int,required=True);evaluation_arguments(p);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);groups,excluded=prepare(a.root)
    if a.limit_groups:groups=groups[:a.limit_groups]
    rows=[q for g in groups for q in g['questions']]
    (a.out/'manifest.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(a.out/'excluded.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in excluded))
    write(a.out/'run.json',capture({'backend':'hf','numerics':a.numerics,'dtype':a.dtype,'device_map':a.device_map,'mode':a.mode,'projection':a.projection,'prefix_cache':a.prefix_cache,'question_batch_size':a.question_batch_size,'max_input_tokens':4000,'context':'Inspect the supplied media carefully.','video_options':{'fps':1,'min_pixels':3136,'max_pixels':50176,'max_frames':32}},a.model_revision,a.model))
    engine=HFMJevEngine(a.model,device_map=a.device_map,dtype=a.dtype,numerics=a.numerics,max_input_tokens=4000);predictions=[];times=[]
    with (a.out/'raw.jsonl').open('w') as f:
        for i,g in enumerate(groups):
            try:
                start=time.perf_counter()
                result=engine.score_many(str(a.root/g['media_path']),[{'id':q['id'],'question':q['question'],'candidates':q['candidates']} for q in g['questions']],modality=g['modality'],mode=a.mode,projection=a.projection,batch_size=a.question_batch_size,use_prefix_cache=a.prefix_cache,video_options={'fps':1,'min_pixels':3136,'max_pixels':50176,'max_frames':32} if g['modality']!='audio' else None)
                elapsed=time.perf_counter()-start;times.append(elapsed)
                for r in result:predictions.append({'id':r['id'],'logits':{c['label']:c['raw_logit'] for c in r['candidates']},'prediction':r['decision']['label']})
                f.write(json.dumps({'group':i,'client_seconds':elapsed,'results':result})+'\n');f.flush()
                (a.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            except Exception as e:
                (a.out/'ERROR.json').write_text(json.dumps({'group':i,'error':repr(e)}));raise
    report={'evaluation':evaluate(rows,predictions),'request_seconds_mean':statistics.mean(times),'native_clotho_excluded':len(excluded),'backend':'transformers','mode':a.mode,'question_execution':{'batch_size':a.question_batch_size,'prefix_cache':a.prefix_cache,'numerics':a.numerics,'projection':a.projection}}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));(a.out/'_SUCCESS').write_text('All selected eligible MCQ rows completed. Native Clotho exclusion retained.\n')
if __name__=='__main__':main()
