"""Seeded BF16 attention dispatch witness on every assigned GPU."""
import json
import torch
import torch.nn.functional as F


def main():
    results=[]
    for device in range(torch.cuda.device_count()):
        torch.manual_seed(7)
        q=torch.randn(1,4,16,64,device=f'cuda:{device}',dtype=torch.bfloat16)
        out=F.scaled_dot_product_attention(q,q,q,is_causal=True)
        assert out.shape==q.shape and torch.isfinite(out).all()
        results.append({'gpu':device,'name':torch.cuda.get_device_name(device),'shape':list(out.shape)})
    assert len(results)==4
    print('AV_KERNEL_WITNESS_OK',json.dumps(results))

if __name__=='__main__':main()
