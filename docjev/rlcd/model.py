"""Native Qwen3-VL processor/template/positions/LM head. No generation or vLLM."""
import hashlib
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from mjev.prompt import PromptBuilder
from .common import sha256


def load_model(path, *, dtype, device=None):
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, attn_implementation='sdpa', local_files_only=True,
        output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError(f'Checkpoint load not exact: {info}')
    # Transformers 5 returns sets for some loading-info fields.
    info = {key: sorted(value) if isinstance(value, set) else value
            for key, value in info.items()}
    if model.config.model_type != 'qwen3_vl':
        raise ValueError('This prototype trains the Qwen3-VL family only')
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if device is not None:
        model.to(device)
    return model, info


class Inputs:
    def __init__(self, model_path, data_root, config):
        self.root = data_root
        self.builder = PromptBuilder(model_path)
        self.processor = self.builder.processor
        self.processor.image_processor.size = {
            'shortest_edge': config['min_pixels'], 'longest_edge': config['max_pixels']}
        self.config = config

    def prepare(self, row):
        from pathlib import Path
        image_path = (Path(self.root) / row['media_path']).resolve()
        if sha256(image_path) != row['image_sha256']:
            raise ValueError('Frozen image content changed')
        candidates = list(row['candidates'].values())
        spec = self.builder.build(str(image_path), self.config['context'],
                                  row['question'], candidates, mode='causal')
        with Image.open(image_path) as image:
            batch = self.processor(text=spec['text'], images=[image.convert('RGB')],
                                   return_tensors='pt', truncation=False, padding=False)
        tokens = batch['input_ids'].shape[-1]
        if tokens > self.config['max_input_tokens']:
            raise ValueError(f'{row["id"]}: {tokens} exceeds input limit; no truncation')
        delta = tokens - len(spec['raw_ids'])
        boundary = spec['spans'][0][0]
        if batch['input_ids'][0, boundary + delta:].tolist() != spec['raw_ids'][boundary:]:
            raise ValueError('Native processor changed candidate suffix')
        if list(row['candidates']) != spec['labels']:
            raise ValueError('Candidate label order changed')
        hashes = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError('Unexpected processor output')
            hashes[key] = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return batch, {'label_ids': spec['label_ids'], 'labels': spec['labels'],
                       'input_tokens': tokens, 'tensor_hashes': hashes,
                       'prompt_sha256': hashlib.sha256(spec['text'].encode()).hexdigest()}


def move_batch(batch, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if k == 'pixel_values' else v.dtype)
            for k, v in batch.items()}


def candidate_logits(model, batch, label_ids):
    output = model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
    labels = torch.tensor(label_ids, device=output.logits.device)
    return output.logits[0, -1].index_select(0, labels).float()


def tensor_digest(items):
    h = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def visual_digest(model):
    return tensor_digest(model.model.visual.state_dict().items())


def position_buffer_digest(model):
    return tensor_digest((name, value) for name, value in model.named_buffers()
                         if name.endswith('inv_freq'))


def cast_parameters_preserve_buffers(module, *, device, dtype):
    """Preserve official FP32 RoPE frequencies when casting frozen parameters."""
    module.to(device=device)
    for parameter in module.parameters():
        parameter.data = parameter.data.to(dtype=dtype)
