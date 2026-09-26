import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('runner',Path(__file__).resolve().parents[2]/'benchmarks/integrated/evaluate_service.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def test_payload_does_not_leak_labels_or_references():
    g={'modality':'audio_video','media_path':'media/test.mp4','questions':[{'id':'q','question':'What sound?','candidates':{'A':'rain','B':'wind'},'label':'B','references':['wind'],'original':{'correct':'wind'},'evidence_start':10}]}
    p=runner.payload(g,'causal')
    assert p['use_audio_in_video'] is True
    assert set(p['questions'][0])=={'id','instructions','criteria'}
    assert p['media'][0]['type']=='video_url'
