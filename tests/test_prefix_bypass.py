"""Request-level bypass must survive the engine adapter."""
import asyncio
from types import SimpleNamespace as N
import pytest,torch
pytest.importorskip("vllm", reason="requires optional pinned vLLM environment")
from mjev.engine import MJevEngine

@pytest.mark.parametrize('requested,debug,expected',[(False,False,False),(True,False,True),(False,True,True)])
def test_cache_bypass_preserved_without_mutating_input(requested,debug,expected):
    observed=[]
    class FakeEngine:
        async def encode(self,prompt,params,request_id):
            observed.append(params.skip_reading_prefix_cache)
            # Debug adds one hidden feature per candidate.
            values=[1.,0.,0.]+([2.,3.] if debug else [])
            yield N(outputs=N(data=torch.tensor(values)),prompt_token_ids=[1,2,3])
    engine=MJevEngine.__new__(MJevEngine);engine.engine=FakeEngine();engine.model_family='qwen3_omni_moe'
    params=N(skip_reading_prefix_cache=requested,extra_kwargs={'mjev':{
        'raw_ids':[1,2,3],'spans':[[0,1],[1,2]],'mode':'causal','debug':False}})
    prepared=({'multi_modal_data':{}},params,['A','B'],[10,11])
    asyncio.run(engine.score_prepared(prepared,'Question',['yes','no'],mode='causal',debug=debug))
    assert observed==[expected]
    assert params.skip_reading_prefix_cache==requested
    assert params.extra_kwargs['mjev']['debug'] is False
