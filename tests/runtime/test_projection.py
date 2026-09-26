from types import SimpleNamespace as N
import torch
import pytest
pytest.importorskip("vllm", reason="runtime projection imports vLLM distributed helpers")
from mjev_vllm_patch import _selective_tp_logits

def test_selective_matches_full_fp32_projection():
    torch.manual_seed(37)
    weight=torch.randn(73,32,dtype=torch.bfloat16)
    hidden=torch.randn(5,32,dtype=torch.bfloat16)
    quant=type('UnquantizedEmbeddingMethod',(),{})()
    head=N(weight=weight,bias=None,quant_method=quant,tp_size=1,
           shard_indices=N(org_vocab_start_index=0,org_vocab_end_index=73))
    model=N(lm_head=head,logits_processor=N(scale=1.,soft_cap=None))
    ids=(3,42,7,71)
    actual=_selective_tp_logits(model,hidden,ids)
    expected=(hidden.float()@weight.float().T)[:,list(ids)]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
