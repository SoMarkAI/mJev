"""Strict mJev-style JSONL loader and ground-truth-free model input adapters."""
import json
from pathlib import Path


def validate(record):
    required = {'id', 'dataset', 'source', 'task', 'modality', 'media_path',
                'media', 'question', 'candidates', 'label'}
    if required - record.keys():
        raise ValueError(f"Missing fields: {required - record.keys()}")
    if record['modality'] not in ('audio', 'video', 'audio_video', 'image'):
        raise ValueError('Unknown modality')
    choices = record['candidates']
    if not isinstance(choices, dict) or len(choices) < 2:
        raise ValueError('Candidates must be an ordered label-to-text mapping')
    if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
           for k, v in choices.items()):
        raise ValueError('Invalid candidate labels/text')
    if record['label'] not in choices:
        raise ValueError('Answer not in candidates')
    if not isinstance(record['question'], str) or not record['question'].strip():
        raise ValueError('Question must be nonempty text')
    p = Path(record['media_path'])
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Media paths must be relative to the benchmark root')
    if record['media'].get('path') != record['media_path']:
        raise ValueError('media and media_path disagree')
    return record


class MJevDataset:
    def __init__(self, manifest, media_root=None, require_media=True):
        self.manifest = Path(manifest).resolve()
        self.root = Path(media_root).resolve() if media_root else self.manifest.parent
        self.records = []
        seen = set()
        with self.manifest.open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = validate(json.loads(line))
                    if row['id'] in seen:
                        raise ValueError('Duplicate sample ID')
                    if require_media and not self.media_path(row).is_file():
                        raise FileNotFoundError(row['media_path'])
                    seen.add(row['id'])
                    self.records.append(row)
                except Exception as exc:
                    raise ValueError(f'{self.manifest}:{line_number}: {exc}') from exc

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def media_path(self, row):
        path = (self.root / row['media_path']).resolve()
        if self.root not in path.parents:
            raise ValueError('Media path escapes dataset root')
        return path

    def model_input(self, row):
        """Allowlist: never forward labels, evidence timestamps or raw annotations."""
        return {'id': row['id'], 'modality': row['modality'],
                'media_path': str(self.media_path(row)), 'question': row['question'],
                'candidates': dict(row['candidates'])}

    def qwen_messages(self, row):
        """Official Qwen-style content structure; decoding stays in its processor.

        Full video is passed, including audio when the caller enables
        use_audio_in_video=True. Evidence timestamps are intentionally not used
        for cropping: doing so would leak annotated evidence locations.
        """
        item = self.model_input(row)
        kind = 'audio' if row['modality'] == 'audio' else ('image' if row['modality'] == 'image' else 'video')
        choices = '\n'.join(f'{k}. {v}' for k, v in item['candidates'].items())
        return [{'role': 'user', 'content': [
            {'type': kind, kind: item['media_path']},
            {'type': 'text', 'text': item['question'] + '\n' + choices
             + '\nSelect the correct candidate and answer with its label.'}]}]
