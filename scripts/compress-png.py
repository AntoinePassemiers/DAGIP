import os
import math
from PIL import Image

import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, '..', 'figures')
os.makedirs(os.path.join(OUT_DIR, 'compressed'), exist_ok=True)

Image.MAX_IMAGE_PIXELS = None


def compress(in_filepath: str, out_filepath: str) -> None:
    img = Image.open(in_filepath)

    total_size = img.size[0] * img.size[1]
    if total_size >= 40000000:
        alpha = math.sqrt((40000000 - 1) / total_size)
        new_size = (int(math.floor(alpha * img.size[0])), int(math.floor(alpha * img.size[1])))
        img.thumbnail(new_size, Image.Resampling.LANCZOS)

    img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
    img.save(out_filepath, optimize=True)


for filename in tqdm.tqdm(os.listdir(OUT_DIR)):
    if filename in ['DAGIP-Illustration.png', 'DAGIP-Figure-ichorCNA.png', 'DAGIP-Figure-Pairs.png']:
        continue
    in_filepath = os.path.join(OUT_DIR, filename)
    out_filepath = os.path.join(OUT_DIR, 'compressed', filename)
    if not os.path.isfile(in_filepath):
        continue

    compress(in_filepath, out_filepath)
