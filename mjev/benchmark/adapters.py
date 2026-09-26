"""Output bridges for raw Qwen/mJev candidate scores.

These functions do not instantiate a model or pretend the image-only mJev
0.1 engine already implements audio/video attention. Use model_input() or
qwen_messages() as the input contract of an AV-capable inference backend.
"""


def from_mjev_result(payload, result):
    expected=payload['candidates']
    rows=result['candidates']
    if len(rows)!=len(expected):raise ValueError('Backend candidate count mismatch')
    scores={}
    for (label,text),candidate in zip(expected.items(),rows):
        if candidate['label']!=label or candidate['text']!=text:
            raise ValueError('Backend changed candidate order, labels or text')
        scores[label]=float(candidate['raw_logit'])
    return {'id':payload['id'],'logits':scores}


def from_qwen_logits(payload, tokenizer, rendered_prompt, next_token_logits):
    """Read original LM logits only after exact contextual single-token checks.

    rendered_prompt must be the exact official chat template + assistant
    prefill used by the forward pass (e.g. ending in Answer:\\n). The vector
    must be the raw vocabulary logits for its next-token position.
    """
    prefix=tokenizer.encode(rendered_prompt,add_special_tokens=False)
    ids={}
    for label in payload['candidates']:
        appended=tokenizer.encode(rendered_prompt+label,add_special_tokens=False)
        if len(appended)!=len(prefix)+1 or appended[:-1]!=prefix:
            raise ValueError(f'Label {label!r} is not one contextual token')
        ids[label]=appended[-1]
    if len(set(ids.values()))!=len(ids):raise ValueError('Candidate token collision')
    return {'id':payload['id'],'logits':{label:float(next_token_logits[token]) for label,token in ids.items()}}


def qwen_vllm_input(dataset, row, processor, *, use_audio_in_video, video_options=None):
    """Build official Qwen AV processor input, without loading model weights.

    video_options are explicitly selected inference settings, not dataset edits.
    The caller must record them in run configuration. For example nframes=4 is
    suitable for a preprocessing smoke test, not a claim of full-video coverage.
    """
    from qwen_omni_utils import process_mm_info
    messages=dataset.qwen_messages(row)
    item=messages[0]['content'][0]
    if video_options:
        allowed={'fps','nframes','min_frames','max_frames','min_pixels','max_pixels',
                 'resized_height','resized_width'}
        if item['type']!='video' or set(video_options)-allowed:
            raise ValueError('Unsupported video processing options')
        item.update(video_options)
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+'Answer:\n'
    prefix=processor.tokenizer.encode(rendered,add_special_tokens=False)
    token_ids={}
    for label in row['candidates']:
        appended=processor.tokenizer.encode(rendered+label,add_special_tokens=False)
        if len(appended)!=len(prefix)+1 or appended[:-1]!=prefix:
            raise ValueError(f'Label {label!r} is not one contextual token')
        token_ids[label]=appended[-1]
    if len(set(token_ids.values()))!=len(token_ids):raise ValueError('Candidate token collision')
    audios,images,videos,video_kwargs=process_mm_info(
        messages,use_audio_in_video=use_audio_in_video,return_video_kwargs=True,
        image_patch_size=processor.image_processor.patch_size)
    video_kwargs=dict(video_kwargs)
    fps=video_kwargs.get('fps')
    if isinstance(fps,list):
        if not fps:video_kwargs.pop('fps')
        elif len(fps)==1:video_kwargs['fps']=float(fps[0])
        else:raise ValueError('Benchmark records contain exactly one video')
    media={}
    for name,value in [('audio',audios),('image',images),('video',videos)]:
        if value is not None:media[name]=value
    prompt={'prompt':rendered,'multi_modal_data':media,
            'mm_processor_kwargs':{'use_audio_in_video':use_audio_in_video,**video_kwargs}}
    return prompt, token_ids
