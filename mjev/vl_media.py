"""Native Qwen3-VL image/video processing, including sampled frame timestamps."""
from copy import deepcopy
from functools import partial
from transformers.image_utils import load_image
from transformers.video_utils import load_video


def prepare_media(processor, media, modality, video_options=None):
    if modality == 'image':
        return {'images': [load_image(media)]}
    if modality != 'video':
        raise ValueError('Qwen3-VL accepts image/video only')
    options = dict(video_options or {})
    sampler = deepcopy(processor.video_processor)
    for key in ('min_frames', 'max_frames'):
        if key in options:
            setattr(sampler, key, options[key])
    count = options.get('nframes')
    fps = options.get('fps', None if count is not None else 1)
    frames, metadata = load_video(media, backend='torchcodec',
        sample_indices_fn=partial(sampler.sample_frames, num_frames=count, fps=fps))
    size = dict(processor.video_processor.size)
    if 'min_pixels' in options:
        size['shortest_edge'] = options['min_pixels']
    if 'max_pixels' in options:
        size['longest_edge'] = options['max_pixels']
    return {'videos': [frames], 'videos_kwargs': {
        'video_metadata': [metadata], 'do_sample_frames': False, 'size': size}}
