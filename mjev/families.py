"""Small model-family boundary; native Transformers modules remain unchanged."""
from transformers import AutoConfig

OMNI = 'qwen3_omni_moe'
VL = 'qwen3_vl'


def family(config):
    kind = config.model_type
    if kind in (OMNI, 'qwen3_omni_moe_thinker'):
        return OMNI
    if kind == VL:
        return VL
    raise ValueError(f'Unsupported model_type {kind!r}; expected Qwen3-Omni or Qwen3-VL')


def detect(path):
    return family(AutoConfig.from_pretrained(path, local_files_only=True))


def text_model(model):
    return model.model.language_model if family(model.config) == VL else model.model


def multimodal_model(model):
    return model.model if family(model.config) == VL else model


def check_modality(kind, modality):
    supported = ('image', 'video') if kind == VL else ('image', 'video', 'audio', 'audio_video')
    if modality not in supported:
        raise ValueError(f'{kind} supports {supported}, not {modality!r}; audio is never silently discarded')


def positions(model, batch):
    device = model.get_input_embeddings().weight.device
    def get(key):
        value = batch.get(key)
        return value.to(device) if hasattr(value, 'to') else value
    if family(model.config) == VL:
        import torch
        types = get('mm_token_type_ids')
        if types is None:
            types = torch.zeros_like(get('input_ids'))
        return model.model.get_rope_index(
            get('input_ids'), types, image_grid_thw=get('image_grid_thw'),
            video_grid_thw=get('video_grid_thw'), attention_mask=get('attention_mask'))[0]
    lengths = get('feature_attention_mask')
    if lengths is not None:
        lengths = lengths.sum(-1)
    return model.get_rope_index(get('input_ids'), get('image_grid_thw'),
        get('video_grid_thw'), get('attention_mask'), batch.get('use_audio_in_video'),
        lengths, get('video_second_per_grid'))[0]
