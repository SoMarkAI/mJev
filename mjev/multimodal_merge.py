"""Typed audio/vision placement for packed requests, including text gaps.

Omni vision tensors retain main + deepstack features; audio tensors contain
only the text hidden width. Use that native type information before splitting.
Never infer modality from token counts or interleaving across a whole batch.
"""
import torch


def merge_audio_vision(inputs, embeddings, vision_mask, audio_mask, levels):
    hidden=inputs.shape[-1]
    vision=[];audio=[];deep=[]
    for value in embeddings:
        if value.ndim!=2:
            raise ValueError('Expected flat native media embeddings')
        if value.shape[-1]==hidden*(levels+1) and levels>0:
            main,extra=value.split([hidden,hidden*levels],dim=-1)
            vision.append(main);deep.append(extra)
        elif value.shape[-1]==hidden:
            audio.append(value)
        else:
            raise ValueError('Unrecognized native media feature width')
    out=inputs.clone()
    stacks=inputs.new_zeros((inputs.shape[0],hidden*levels))
    for mask,values in [(vision_mask,vision),(audio_mask,audio)]:
        if sum(v.shape[0] for v in values)!=int(mask.sum()):
            raise ValueError('Media embeddings do not match typed token positions')
        if values:out[mask.to(out.device)]=torch.cat(values).to(out)
    if deep:stacks[vision_mask.to(stacks.device)]=torch.cat(deep).to(stacks)
    return out,stacks.view(inputs.shape[0],levels,hidden).permute(1,0,2).contiguous()
