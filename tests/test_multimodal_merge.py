import pytest
import torch
from mjev.multimodal_merge import merge_audio_vision

@pytest.mark.parametrize('audio_first',[False,True])
def test_packed_requests_keep_typed_media_and_deepstack(audio_first):
    # Two interleaved requests separated by ordinary question tokens.
    ids=torch.tensor([9,1,2,1,2,9,9,1,2,1,2,9])
    inputs=torch.full((12,2),-1.)
    v1=torch.tensor([[10.,11.,110.,111.],[12.,13.,112.,113.]])
    v2=v1+20;a1=torch.tensor([[50.,51.],[52.,53.]]);a2=a1+20
    values=[a1,v1,a2,v2] if audio_first else [v1,a1,v2,a2]
    before=[x.clone() for x in values]
    out,deep=merge_audio_vision(inputs,values,ids==1,ids==2,1)
    torch.testing.assert_close(out[ids==1],torch.cat([v1[:,:2],v2[:,:2]]))
    torch.testing.assert_close(out[ids==2],torch.cat([a1,a2]))
    torch.testing.assert_close(deep[0,ids==1],torch.cat([v1[:,2:],v2[:,2:]]))
    assert (out[ids==9]==-1).all() and (deep[0,ids!=1]==0).all()
    for a,b in zip(values,before):assert torch.equal(a,b)

def test_reject_wrong_modality_counts():
    with pytest.raises(ValueError,match='typed token'):
        merge_audio_vision(torch.zeros(4,2),[torch.zeros(2,4),torch.zeros(2,2)],
                           torch.tensor([1,0,0,0],dtype=torch.bool),
                           torch.tensor([0,1,1,1],dtype=torch.bool),1)
