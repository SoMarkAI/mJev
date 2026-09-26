import asyncio
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from mjev.benchmark.dataset import MJevDataset, validate
from mjev.benchmark.evaluate import evaluate
from mjev.benchmark.run import run


def row(id='x', label='B', choices=None):
    return {'id':id,'dataset':'fixture','source':{'repository':'synthetic'},'task':['test'],
            'modality':'audio','media_path':'media/a.wav','media':{'type':'audio','path':'media/a.wav'},
            'question':'Which?', 'candidates':choices or {'A':'first','B':'second'},'label':label,
            'original':{'correct_answer':label},'metadata':{'evidence_start':'00:01'}}


class MetricsTests(unittest.TestCase):
    def test_known_metrics(self):
        rows=[row('x'),row('y',label='A')]
        pred=[{'id':'x','probabilities':{'B':0.75,'A':0.25}},
              {'id':'y','probabilities':{'A':0.5,'B':0.5}}]
        result=evaluate(rows,pred,bins=2)['metrics']
        self.assertEqual(result['accuracy'],1)
        self.assertAlmostEqual(result['nll'],(-math.log(.75)-math.log(.5))/2)
        self.assertAlmostEqual(result['brier_score'],(.125+.5)/2)
        self.assertAlmostEqual(result['ece'],.375)
    def test_logits_stability_and_dynamic_counts(self):
        r=row(choices={'A':'a','B':'b','C':'c'})
        m=evaluate([r],[{'id':'x','logits':{'A':1000,'B':999,'C':998}}])['metrics']
        self.assertAlmostEqual(m['nll'],1+math.log(1+math.exp(-1)+math.exp(-2)))
        self.assertEqual(m['accuracy'],0)
    def test_zero_probability_not_hidden(self):
        m=evaluate([row()],[{'id':'x','probabilities':{'A':1,'B':0}}])['metrics']
        self.assertIsNone(m['nll']);self.assertTrue(m['nll_is_infinite'])
        self.assertEqual(m['brier_score'],2)
        self.assertEqual(m['ece'],1)
        json.dumps(m,allow_nan=False)
    def test_missing_duplicate_unknown(self):
        for pred in [[],[{'id':'other','logits':{'A':1,'B':2}}],
                     [{'id':'x','logits':{'A':1,'B':2}}]*2]:
            with self.assertRaises(ValueError):evaluate([row()],pred)
        self.assertEqual(evaluate([row()],[],allow_partial=True)['coverage'],0)
    def test_invalid_distributions(self):
        for values in [{'A':.5,'B':.6},{'A':float('nan'),'B':1},{'A':-.1,'B':1.1},{'A':1}, {'A':True,'B':False}]:
            with self.assertRaises(ValueError):evaluate([row()],[{'id':'x','probabilities':values}])
    def test_label_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate([row()],[{'id':'x','logits':{'A':1,'B':2},'prediction':'A'}])


class LoaderRunnerTests(unittest.TestCase):
    def test_inputs_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'media').mkdir();(root/'media/a.wav').touch()
            (root/'manifest.jsonl').write_text(json.dumps(row())+'\n')
            ds=MJevDataset(root/'manifest.jsonl')
            self.assertEqual(set(ds.model_input(ds.records[0])),{'id','modality','media_path','question','candidates'})
            message=ds.qwen_messages(ds.records[0]);self.assertNotIn('evidence',json.dumps(message))
            calls=[]
            async def score(payload):
                calls.append(payload);return {'logits':{'A':1,'B':2}}
            output=root/'predictions.jsonl'
            asyncio.run(run(ds,score,output));asyncio.run(run(ds,score,output))
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):
                asyncio.run(run(ds,score,output,run_config={'model':'different'}))
            self.assertEqual(evaluate(ds,[json.loads(output.read_text())])['metrics']['accuracy'],1)
    def test_path_escape(self):
        r=row();r['media_path']='../secret'
        with self.assertRaises(ValueError):validate(r)


class ConversionTests(unittest.TestCase):
    def test_option_text_preserved_and_sampling_stable(self):
        from mjev.benchmark.prepare import parse_options,pick
        text='A. First.\nB. Multi\nline.\nC. Last. '
        self.assertEqual(parse_options(text),{'A':'First.','B':'Multi\nline.','C':'Last. '})
        self.assertEqual(pick(list('abcd'),2,10,'x',str),pick(list('dcba'),2,10,'x',str))


class SelectiveZipTests(unittest.TestCase):
    def test_selected_only_and_crc(self):
        from mjev.benchmark.download import extract_selected_zip
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('videos/001.mp4',b'chosen'*100)
            z.writestr('videos/002.mp4',b'not-chosen'*100)
        class FakeRange(io.BytesIO):
            transferred=0
        with tempfile.TemporaryDirectory() as d, patch('mjev.benchmark.download.HTTPRangeFile',lambda url:FakeRange(buffer.getvalue())):
            dst=Path(d)/'001.mp4'
            result=extract_selected_zip('https://example.test/file.zip',{'001.mp4':dst})
            self.assertEqual(dst.read_bytes(),b'chosen'*100)
            self.assertFalse((Path(d)/'002.mp4').exists())
            self.assertEqual(result[0]['bytes'],600)


class AdapterTests(unittest.TestCase):
    def test_omni_bridge_rejects_reordering(self):
        from mjev.benchmark.adapters import from_mjev_result
        payload={'id':'x','candidates':{'A':'first','B':'second'}}
        result={'candidates':[{'label':'A','text':'first','raw_logit':1},{'label':'B','text':'second','raw_logit':2}]}
        self.assertEqual(from_mjev_result(payload,result)['logits'],{'A':1.,'B':2.})
        result['candidates'].reverse()
        with self.assertRaises(ValueError):from_mjev_result(payload,result)
    def test_contextual_token_check(self):
        from mjev.benchmark.adapters import from_qwen_logits
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return [ord(x) for x in text]
        p={'id':'x','candidates':{'A':'one','B':'two'}}
        self.assertEqual(from_qwen_logits(p,Tokenizer(),'Answer:\n',list(range(100)))['logits'],{'A':65.,'B':66.})
        p['candidates']={'AA':'one','B':'two'}
        with self.assertRaises(ValueError):from_qwen_logits(p,Tokenizer(),'Answer:\n',list(range(100)))

class FailureTests(unittest.TestCase):
    def test_scorer_fail_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'media').mkdir();(root/'media/a.wav').touch()
            (root/'manifest.jsonl').write_text('\n'.join(json.dumps(row(str(i))) for i in range(4)))
            calls=[]
            async def fail(payload):
                calls.append(payload['id']);raise RuntimeError('quota')
            with self.assertRaises(RuntimeError):asyncio.run(run(MJevDataset(root/'manifest.jsonl'),fail,root/'pred.jsonl',1))
            self.assertEqual(len(calls),1)
    def test_range_200_refused(self):
        from mjev.benchmark.download import HTTPRangeFile
        class Response:
            status_code=200;headers={}
            def raise_for_status(self):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            @property
            def content(self):raise AssertionError('Must not read full ZIP body')
        with patch('requests.Session.get',return_value=Response()):
            with self.assertRaises(ValueError):HTTPRangeFile('https://example.test/full.zip')


class ProcessorBridgeTests(unittest.TestCase):
    def test_patch_size_and_scalar_fps(self):
        import types
        from unittest.mock import Mock
        from mjev.benchmark.adapters import qwen_vllm_input
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return list(text.encode())
        processor=types.SimpleNamespace(tokenizer=Tokenizer(),image_processor=types.SimpleNamespace(patch_size=16),
                                        apply_chat_template=lambda *a,**k:'Rendered prompt\n')
        dataset=types.SimpleNamespace(qwen_messages=lambda r:[{'role':'user','content':[{'type':'video','video':'local.mp4'}]}])
        helper=Mock(return_value=(None,None,None,{'fps':[]}))
        with patch.dict('sys.modules',{'qwen_omni_utils':types.SimpleNamespace(process_mm_info=helper)}):
            prompt,_=qwen_vllm_input(dataset,row(),processor,use_audio_in_video=False)
            self.assertNotIn('fps',prompt['mm_processor_kwargs'])
            self.assertEqual(helper.call_args.kwargs['image_patch_size'],16)
            helper.return_value=(None,None,['frames'],{'fps':[.5]})
            prompt,_=qwen_vllm_input(dataset,row(),processor,use_audio_in_video=True)
            self.assertEqual(prompt['mm_processor_kwargs']['fps'],.5)

class QualityTests(unittest.TestCase):
    def test_duration_warnings_preserve_reference(self):
        from mjev.benchmark.audit import duration_warnings
        a=row('x');a['metadata']['video_duration']=100
        b=row('y');b['metadata']['video_duration']=100
        self.assertEqual(duration_warnings([a],[{'media_path':a['media_path'],'duration_seconds':101}]),[])
        warnings=duration_warnings([a,b],[{'media_path':a['media_path'],'duration_seconds':105}])
        self.assertEqual(warnings[0]['question_ids'],['x','y'])
        self.assertEqual(a['metadata']['video_duration'],100)

if __name__=='__main__':unittest.main()
