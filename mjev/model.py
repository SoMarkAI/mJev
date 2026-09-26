import torch
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

class LMReadout(Pooler):
    """Parameter-free adapter; calls the existing, unchanged LM Head."""
    def __init__(self, owner):
        super().__init__()
        # Avoid registering the parent as a child module / duplicating weights.
        object.__setattr__(self, '_owner', owner)
    def get_supported_tasks(self):
        return {'classify'}
    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        last = hidden_states[cursor.last_token_indices_gpu]
        logits = self._owner.compute_logits(last)
        result = []
        for i, params in enumerate(pooling_metadata.pooling_params):
            spec = (params.extra_kwargs or {}).get('mjev')
            if spec is None:  # vLLM's own memory profiling dummy requests
                result.append(logits[i, :2].float())
                continue
            from .patch import _ACTIVE
            active = _ACTIVE.get()
            if spec['mode'] != 'stock':
                req_id = active[0].input_batch.req_ids[i] if active else None
                layers = active[1].get('custom_layers', {}).get(req_id, set()) if active else set()
                if len(layers) != len(self._owner.language_model.model.layers):
                    raise RuntimeError('Not every request decoder layer executed the candidate mask')
            selected = logits[i, spec['label_ids']].float()
            cached = int(cursor.seq_lens_cpu[i] - cursor.num_scheduled_tokens_cpu[i])
            pieces = [selected, selected.new_tensor([cached])]
            if spec.get('debug'):
                # Debug requests disable prefix reads, and must fit one prefill.
                if cached != 0:
                    raise RuntimeError('Debug hidden probes require full prefill')
                offset = int(cursor.first_token_indices_gpu[i])
                probes = [b - 1 + offset for a, b in spec['spans']]
                pieces.append(hidden_states[probes].float().flatten())
            result.append(torch.cat(pieces))
        return result

class MJevProcessor(Qwen3OmniMoeThinkerMultiModalProcessor):
    """Preserve audio placeholders when HF already expanded interleaved video.

    vLLM 0.25.1 derives them only in the non-expanded branch; newer HF
    processors also return expanded video/audio text on this path.
    """
    def _maybe_apply_prompt_updates(self, mm_items, prompt_ids, mm_kwargs,
                                    mm_prompt_updates, is_update_applied):
        interleaved=any(item is not None and bool(item['use_audio_in_video'].data)
                        for item in mm_kwargs.get('video', []))
        if is_update_applied and interleaved:
            counts=mm_items.get_all_counts()
            self._validate_mm_kwargs(mm_kwargs,counts)
            # HF and vLLM can interleave chunks differently even with the same
            # counts. Locate the already-expanded native region, never regenerate
            # its token order from vLLM's older replacement template.
            from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
            config=self.info.get_hf_config()
            starts=[i for i,t in enumerate(prompt_ids) if t==config.vision_start_token_id]
            ends=[i for i,t in enumerate(prompt_ids) if t==config.vision_end_token_id]
            if counts.get('video')!=1 or counts.get('audio')!=1 or len(starts)!=1 or len(ends)!=1 or starts[0]>=ends[0]:
                raise ValueError('Expanded interleaved path requires exactly one video/audio pair')
            start,end=starts[0]+1,ends[0]
            tokens=prompt_ids[start:end]
            placeholders={}
            for modality,token_id in [('video',config.video_token_id),('audio',config.audio_token_id)]:
                mask=torch.tensor(tokens)==token_id
                if not mask.any():raise ValueError(f'Expanded video is missing {modality} features')
                placeholders[modality]=[PlaceholderFeaturesInfo(modality=modality,item_idx=0,
                    start_idx=start,tokens=tokens,is_embed=mask)]
            self._validate_mm_placeholders(placeholders,counts)
            return prompt_ids,placeholders
        return super()._maybe_apply_prompt_updates(mm_items,prompt_ids,mm_kwargs,
                                                   mm_prompt_updates,is_update_applied)


@MULTIMODAL_REGISTRY.register_processor(
    MJevProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class MJevThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
    is_pooling_model = True
    default_seq_pooling_type = 'LAST'
    default_tok_pooling_type = 'ALL'
    attn_type = 'decoder'
    def embed_input_ids(self,input_ids,multimodal_embeddings=None,*,is_multimodal=None):
        # The upstream whole-batch interleaving heuristic fails when two
        # audio-video requests are separated by their question/text tokens.
        # Retained deepstack width identifies vision independently of order.
        levels=len(self.visual.deepstack_visual_indexes or [])
        if multimodal_embeddings is not None and len(multimodal_embeddings)>0 and is_multimodal is not None and levels:
            ids=input_ids.to(is_multimodal.device)
            audio=is_multimodal & (ids==self.config.audio_token_id)
            vision=is_multimodal & ((ids==self.config.video_token_id) | (ids==self.config.image_token_id))
            if bool(audio.any()) and bool(vision.any()):
                if not torch.equal(audio|vision,is_multimodal):
                    raise ValueError('Unsupported media token in mixed audio/vision batch')
                from .multimodal_merge import merge_audio_vision
                inputs=self._embed_text_input_ids(input_ids,self.language_model.embed_input_ids,
                                                  is_multimodal=is_multimodal)
                result,deep=merge_audio_vision(inputs,multimodal_embeddings,vision,audio,levels)
                self._set_deepstack_input_embeds(deep)
                return result
        return super().embed_input_ids(input_ids,multimodal_embeddings,is_multimodal=is_multimodal)

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        lp = self.language_model.logits_processor
        assert lp.scale == 1.0 and lp.soft_cap is None
        # Pooling runs on each TP rank. Gather the unchanged vocabulary logits.
        lp.use_all_gather = True
        self.pooler = LMReadout(self)
        assert sum(p.numel() for p in self.pooler.parameters()) == 0


from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration, Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo, Qwen3VLDummyInputsBuilder,
)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class MJevVL(Qwen3VLForConditionalGeneration):
    """Native Qwen3-VL encoders/DeepStack with the shared parameter-free readout."""
    is_pooling_model = True
    default_seq_pooling_type = 'LAST'
    default_tok_pooling_type = 'ALL'
    attn_type = 'decoder'

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        lp = self.language_model.logits_processor
        assert lp.scale == 1.0 and lp.soft_cap is None
        lp.use_all_gather = True
        self.pooler = LMReadout(self)
        assert sum(p.numel() for p in self.pooler.parameters()) == 0
