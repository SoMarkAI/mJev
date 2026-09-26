"""Shared, model-independent validation for CLI and scoring inputs."""
from pathlib import Path


def candidate_labels(count):
    labels = []
    for i in range(count):
        label, value = '', i + 1
        while value:
            value, remainder = divmod(value - 1, 26)
            label = chr(65 + remainder) + label
        labels.append(label)
    return labels


def normalize_question(question):
    if not isinstance(question, dict):
        raise ValueError('Each question must be an object')
    text = question.get('question')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Question must be nonempty text')
    candidates = question.get('candidates')
    if isinstance(candidates, dict):
        if list(candidates) != candidate_labels(len(candidates)):
            raise ValueError('Candidate mapping must use ordered A, B, ... labels')
        candidates = list(candidates.values())
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 128:
        raise ValueError('Require a list of 2..128 candidates')
    if not all(isinstance(value, str) and value.strip() for value in candidates):
        raise ValueError('Candidates must be nonempty strings')
    if len(set(candidates)) != len(candidates):
        raise ValueError('Candidates must be distinct nonempty strings')
    return {**question, 'candidates': list(candidates)}


def normalize_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions must be a nonempty list')
    return [normalize_question(question) for question in questions]


def normalize_task(task, base_dir, *, default_context='Inspect the supplied media carefully.'):
    if not isinstance(task, dict):
        raise ValueError('Input must be an object')
    questions = normalize_questions(task['questions'] if 'questions' in task else [task])
    modality = task.get('modality', 'image')
    if modality not in ('image', 'video', 'audio', 'audio_video'):
        raise ValueError('Unsupported media modality')
    media = task.get('media_path', task.get('image'))
    if not isinstance(media, str) or not media.strip():
        raise ValueError('media_path (or image) must be a nonempty path')
    context = task.get('context', default_context)
    if not isinstance(context, str):
        raise ValueError('Context must be text')
    options = task.get('video_options')
    if options is not None:
        allowed = {'fps', 'nframes', 'min_frames', 'max_frames', 'min_pixels', 'max_pixels'}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError('Unsupported video options')
        if options and modality not in ('video', 'audio_video'):
            raise ValueError('Video options require video or audio_video')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float('inf') for v in options.values()):
            raise ValueError('Video options must be finite positive numbers')
    return dict(media=str((Path(base_dir) / media).resolve()), modality=modality,
                context=context, video_options=options, questions=questions)
