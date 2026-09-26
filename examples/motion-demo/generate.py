"""Create the project-owned six-second motion demo (Pillow + system FFmpeg)."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw


def frame(t):
    im = Image.new('RGB', (640, 360), '#f1f5fb')
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 20, 620, 340), radius=24, fill='white')
    d.line((50, 295, 590, 295), fill='#cbd5e1', width=3)
    # First the red ball moves right (1–3 s); then the blue square rises (3–5 s).
    x = 100 + 390 * min(1, max(0, (t-1)/2))
    y = 248 - 155 * min(1, max(0, (t-3)/2))
    d.ellipse((x-27, 298, x+27, 307), fill='#e2e8f0')
    d.ellipse((x-28, 238, x+28, 294), fill='#ef4444', outline='#b91c1c', width=2)
    d.ellipse((x-17, 247, x-5, 259), fill='#fca5a5')
    d.rounded_rectangle((273, y-28, 329, y+28), radius=5, fill='#3b82f6', outline='#1d4ed8', width=2)
    return im


def main():
    root = Path(__file__).resolve().parent
    frames = [frame(i/12) for i in range(72)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(frames): im.save(Path(tmp)/f'{i:03d}.png')
        subprocess.run(['ffmpeg','-y','-v','error','-framerate','12','-i',str(Path(tmp)/'%03d.png'),
                        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'motion.mp4')],check=True)
    frames[0].save(root/'poster.png')
    frames[0].save(root/'preview.gif',save_all=True,append_images=frames[1:],duration=83,loop=0,optimize=True)

if __name__ == '__main__': main()
