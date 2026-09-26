"""Official chat rendering; candidate spans tracked after image expansion."""
from transformers import AutoProcessor

class Protocol:
    def __init__(self, model):
        from .families import detect
        self.model_family=detect(model)
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer

    @staticmethod
    def labels(n):
        result = []
        for i in range(n):
            s, x = '', i + 1
            while x:
                x, rem = divmod(x - 1, 26)
                s = chr(65 + rem) + s
            result.append(s)
        return result

    def build(self, image, context, question, candidates, mode='causal', debug=False,
              *, modality='image', video_options=None):
        from .families import check_modality, VL
        check_modality(self.model_family,modality)
        from .prompt import PromptBuilder
        rendered_spec = PromptBuilder.build(self, image, context, question, candidates, mode, debug,
                                            modality=modality, video_options=video_options)
        rendered, messages, raw, spans, labels, label_ids = [rendered_spec[k] for k in
            ('text', 'messages', 'raw_ids', 'spans', 'labels', 'label_ids')]
        spec = {'raw_ids': raw, 'spans': spans, 'label_ids': label_ids,
                'mode': mode, 'debug': debug}
        from vllm import PoolingParams
        params = PoolingParams(task='classify', use_activation=False,
                              skip_reading_prefix_cache=debug, extra_kwargs={'mjev': spec})
        if modality == 'image':
            if isinstance(image,str):
                from transformers.image_utils import load_image
                image=load_image(image)
            prompt = {'prompt': rendered, 'multi_modal_data': {'image': image}}
        elif self.model_family==VL:
            from dataclasses import asdict
            from .vl_media import prepare_media
            decoded=prepare_media(self.processor,image,modality,video_options)
            options=dict(decoded['videos_kwargs'])
            metadata=asdict(options.pop('video_metadata')[0])
            prompt={'prompt':rendered,'multi_modal_data':{'video':[(decoded['videos'][0],metadata)]},
                    'mm_processor_kwargs':options}
        else:
            from qwen_omni_utils import process_mm_info
            use_audio = modality == 'audio_video'
            audios, images, videos, kwargs = process_mm_info(
                messages, use_audio_in_video=use_audio, return_video_kwargs=True,
                image_patch_size=self.processor.image_processor.patch_size)
            kwargs = dict(kwargs)
            fps = kwargs.get('fps')
            if isinstance(fps, list):
                if not fps:
                    kwargs.pop('fps')
                elif len(fps) == 1:
                    kwargs['fps'] = float(fps[0])
                else:
                    raise ValueError('One video per request is supported')
            prompt = {'prompt': rendered,
                      'multi_modal_data': {k:v for k,v in [('audio',audios), ('image',images), ('video',videos)] if v is not None},
                      'mm_processor_kwargs': {'use_audio_in_video': use_audio, **kwargs}}
        return prompt, params, labels, label_ids
