"""Encode every post through the image pipelines and text templates with the same frozen CLIP ViT-B/32.

Image pipelines (20): framing {CLIP center crop, 90% center crop, 80% center crop} x JPEG {none, q60, q30}
x resolution {original, half}, plus a quarter-resolution thumbnail (JPEG q40) and a WebP copy (q50).
Text templates (5): "{}", "a photo of {}", "a post titled: {}", "an image of {}", "caption: {}".
--switch_targets encodes the pipelines of the benign pipeline upgrade instead.

Writes ROOT/pipe_img_<name>/img.npy and ROOT/pipe_txt_<name>/txt.npy, each with an index.parquet of post ids
(posts whose image is available); data/build_streams.py assigns each identity a unique pair.
"""
import argparse, io, os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

FRAMES, COMPS, RES = ("clip", "c90", "c80"), ("none", "q60", "q30"), ("full", "half")
IMG_PIPES = [f"{f}-{c}-{r}" for f in FRAMES for c in COMPS for r in RES] + ["thumb", "webp"]   # 20
TXT_PIPES = {"plain": "{}", "photo": "a photo of {}", "post": "a post titled: {}",
             "image": "an image of {}", "caption": "caption: {}"}                               # 5
MODEL = "openai/clip-vit-base-patch32"
# benign pipeline upgrade (pairs not used by any identity before the upgrade)
SWITCH_IMG = ["c70-none-full", "c70-q60-half", "c75-q45-full", "c85-none-half", "clip-q45-full"]
SWITCH_TXT = {"picture": "a picture of {}", "news": "news photo: {}", "reddit": "reddit post: {}", "shows": "this shows {}"}

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True); p.add_argument("--images", required=True)
p.add_argument("--out_root", required=True)
p.add_argument("--batch", type=int, default=512); p.add_argument("--workers", type=int, default=48)
p.add_argument("--switch_targets", action="store_true", help="encode the pipeline-upgrade pairs instead")
p.add_argument("--text_col", default="clean_title")
a = p.parse_args()

df = pd.read_parquet(a.manifest)
df["path"] = [os.path.join(a.images, f[:2], f) for f in df["fname"]]
have = np.array([os.path.exists(q) and os.path.getsize(q) > 512 for q in df["path"]])
df = df[have].reset_index(drop=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL).to(dev).eval().half()
proc = CLIPProcessor.from_pretrained(MODEL); ip = proc.image_processor


def _recompress(im, fmt, q):
    buf = io.BytesIO(); im.save(buf, fmt, quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def img_pipe(im, name):
    w, h = im.size
    if name == "thumb":
        im = _recompress(im.resize((max(w // 4, 32), max(h // 4, 32)), Image.BILINEAR), "JPEG", 40)
        return im.resize((w, h), Image.BILINEAR)
    if name == "webp":
        return _recompress(im, "WEBP", 50)
    frame, comp, res = name.split("-")
    if res == "half":
        im = im.resize((max(w // 2, 32), max(h // 2, 32)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if comp != "none":
        im = _recompress(im, "JPEG", int(comp[1:]))
    if frame.startswith("c") and frame[1:].isdigit():          # centre crop keeping cNN percent
        f = (1.0 - int(frame[1:]) / 100.0) / 2.0
        dx, dy = max(int(w * f), 1), max(int(h * f), 1); im = im.crop((dx, dy, w - dx, h - dy))
    return im


class DS(Dataset):
    def __init__(self, paths, name): self.p, self.n = paths, name
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        try:
            im = Image.open(self.p[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224))
        return ip(img_pipe(im, self.n), return_tensors="pt")["pixel_values"][0]


def save(dirname, arr, fname):
    d = os.path.join(a.out_root, dirname); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, fname), arr)
    df[["uid"]].to_parquet(os.path.join(d, "index.parquet"), index=False)


if a.switch_targets:
    IMG_PIPES, TXT_PIPES = SWITCH_IMG, SWITCH_TXT
for name in IMG_PIPES:
    if os.path.exists(os.path.join(a.out_root, f"pipe_img_{name}", "img.npy")):
        continue
    dl = DataLoader(DS(list(df["path"]), name), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    X = np.zeros((len(df), 512), np.float32); i = 0
    with torch.no_grad():
        for b in dl:
            f = model.visual_projection(model.vision_model(pixel_values=b.to(dev).half()).pooler_output)
            X[i:i + len(f)] = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy(); i += len(f)
    save(f"pipe_img_{name}", X, "img.npy")

for name, tpl in TXT_PIPES.items():
    if os.path.exists(os.path.join(a.out_root, f"pipe_txt_{name}", "txt.npy")):
        continue
    texts = [tpl.format(str(t))[:300] for t in df[a.text_col]]
    Y = np.zeros((len(df), 512), np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), a.batch):
            tk = proc.tokenizer(texts[s:s + a.batch], padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(dev)
            f = model.text_projection(model.text_model(**tk).pooler_output)
            Y[s:s + len(f)] = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
    save(f"pipe_txt_{name}", Y, "txt.npy")
