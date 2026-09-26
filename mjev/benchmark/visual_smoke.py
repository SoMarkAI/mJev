"""Generate owned visual fixtures. Synthetic smoke coverage, not benchmark accuracy."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
from .prepare import write_jsonl
from .provenance import sha256, write


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    image=Image.new('RGB',(224,224),'white')
    ImageDraw.Draw(image).rectangle((48,72,176,152),fill='red')
    image.save(root/'rectangle.png')
    subprocess.run(['ffmpeg','-v','error','-loop','1','-i',str(root/'rectangle.png'),
        '-t','2','-r','8','-c:v','libx264','-pix_fmt','yuv420p',str(root/'rectangle.mp4')],check=True)
    questions=[('What color is the rectangle?',{'A':'Red','B':'Blue','C':'Green'},'A'),
        ('Which shape is shown?',{'A':'Circle','B':'Rectangle'},'B'),
        ('What color is the background?',{'A':'Black','B':'Blue','C':'Green','D':'White'},'D')]
    rows=[]
    for modality,media in [('image','rectangle.png'),('video','rectangle.mp4')]:
        for i,(question,choices,label) in enumerate(questions):
            rows.append(dict(id=f'visual-smoke:{modality}:{i}',dataset='mjev-synthetic-smoke',
                source={'generator':'mjev.benchmark.visual_smoke','version':1},task='synthetic-visual-smoke',
                modality=modality,media_path=media,media={'type':modality,'path':media},
                question=question,candidates=choices,label=label,source_dataset='mjev-synthetic-smoke',
                source_id=f'{modality}:{i}',license='Apache-2.0',media_license='Apache-2.0',
                redistribution_allowed=True,source_url='mjev/benchmark/visual_smoke.py'))
    write_jsonl(root/'manifest.jsonl',rows)
    write(root/'receipts.json',{name:sha256(root/name) for name in ['rectangle.png','rectangle.mp4','manifest.jsonl']})
    (root/'_DATA_READY').write_text('Owned synthetic fixture; not an original-label public benchmark.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    prepare(p.parse_args().root)

if __name__=='__main__':main()
