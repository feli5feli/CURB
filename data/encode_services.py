"""Heterogeneous services: encode the posts with each service's own pretrained encoder and input (native preprocessing).

  S1  OpenAI CLIP ViT-B/32      full image + title
  S2  OpenCLIP ViT-B/32 LAION-2B full image + title
  S3  SigLIP-base               full image + title
  S4  MetaCLIP ViT-B/32         full image + title
  S5  ALIGN-base                full image + title
  S6  DINOv2-base + e5-base-v2  full image + title
  S7  CLIP ViT-B/16             full image; its text is its OWN BLIP caption of the image (never sees the title)
  S8  SigLIP-base               quarter-resolution thumbnail + first half of the title's words
  S9  MetaCLIP ViT-B/32         centre 50% crop of the image + title
  S10 ALIGN-base                full image + first five words of the title

Posts 0-11999 and 20000-23999 of the Fakeddit stream (0-3999 fit the alignment maps, 4000-7999 select their
regularisation, 8000-11999 registration, 20000-23999 test). Encoders are read from $CURB_MODELS (S1 from the HF hub).
Writes OUT/emb_<S>.npz (L2-normalised raw img, txt, post index, uid) and OUT/captions.json (S7).
"""
import argparse, json, os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoImageProcessor, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor,
                          AlignModel, AlignProcessor, BlipForConditionalGeneration, BlipProcessor)

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--services", nargs="*", default=[f"S{k}" for k in range(1, 11)])
p.add_argument("--batch", type=int, default=128); p.add_argument("--workers", type=int, default=8)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
M = os.environ.get("CURB_MODELS", "models"); D = os.environ.get("CURB_DATA", "cache")
POSTS = np.r_[np.arange(0, 12000), np.arange(20000, 24000)]
idx = pd.read_parquet(f"{D}/embed_d1/index.parquet"); order = np.argsort(idx["created_utc"].values, kind="mergesort")[:80000]
uid = idx["uid"].astype(str).values[order][POSTS]
man = pd.read_parquet(f"{D}/manifest/d1_manifest.parquet").set_index("uid")
paths = [os.path.join(f"{D}/images", f[:2], f) for f in man.loc[uid, "fname"].values]
titles = [str(t)[:300] for t in man.loc[uid, "clean_title"].values]
dev = "cuda"; nrm = lambda f: torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def load(i):
    try: return Image.open(paths[i]).convert("RGB")
    except Exception: return Image.new("RGB", (224, 224))


def thumb(im):
    w, h = im.size; return im.resize((max(w // 4, 16), max(h // 4, 16)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def crop50(im):
    w, h = im.size; dx, dy = w // 4, h // 4; return im.crop((dx, dy, w - dx, h - dy))


class DS(Dataset):
    def __init__(self, proc, view): self.proc, self.view = proc, view
    def __len__(self): return len(paths)
    def __getitem__(self, i): return self.proc(images=self.view(load(i)), return_tensors="pt")["pixel_values"][0]


def images(proc, fn, view=lambda im: im, dtype=torch.float16):
    out = []
    with torch.no_grad():
        for b in DataLoader(DS(proc, view), batch_size=a.batch, num_workers=a.workers, pin_memory=True): out.append(nrm(fn(b.to(dev).to(dtype))))
    return np.concatenate(out)


def text(tok, fn, texts, max_len=77, pad=True):
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), a.batch):
            t = tok(texts[s:s + a.batch], padding=pad, truncation=True, max_length=max_len, return_tensors="pt").to(dev); out.append(nrm(fn(t)))
    return np.concatenate(out)


clip_txt = lambda mdl: (lambda t: mdl.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"]))
half = lambda t: " ".join(t.split()[:max(1, (len(t.split()) + 1) // 2)])
five = lambda t: " ".join(t.split()[:5])
for S in a.services:
    if S in ("S1", "S2", "S4"):
        src = {"S1": "openai/clip-vit-base-patch32", "S2": f"{M}/CLIP-ViT-B-32-laion2B-s34B-b79K", "S4": f"{M}/metaclip-b32-400m"}[S]
        mdl = CLIPModel.from_pretrained(src).to(dev).eval().half(); pr = CLIPProcessor.from_pretrained(src)
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b))
        Y = text(pr.tokenizer, clip_txt(mdl), titles)
    elif S == "S3":
        src = f"{M}/siglip-base-patch16-224"; mdl = AutoModel.from_pretrained(src).to(dev).eval().half(); pr = AutoProcessor.from_pretrained(src)
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b))
        Y = text(pr.tokenizer, lambda t: mdl.get_text_features(**{k: v for k, v in t.items() if k in ("input_ids", "attention_mask")}),
                 titles, max_len=64, pad="max_length")
    elif S == "S5":
        src = f"{M}/align-base"; mdl = AlignModel.from_pretrained(src).to(dev).eval(); pr = AlignProcessor.from_pretrained(src)   # fp32: fp16 overflows
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b), dtype=torch.float32)
        Y = text(pr.tokenizer, clip_txt(mdl), titles, max_len=64)
    elif S == "S6":
        mi = AutoModel.from_pretrained(f"{M}/dinov2-base").to(dev).eval().half(); pi = AutoImageProcessor.from_pretrained(f"{M}/dinov2-base")
        X = images(pi, lambda b: mi(pixel_values=b).pooler_output); del mi
        mdl = AutoModel.from_pretrained(f"{M}/e5-base-v2").to(dev).eval().half(); tk = AutoTokenizer.from_pretrained(f"{M}/e5-base-v2")
        def e5(t):
            h = mdl(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state.float(); msk = t["attention_mask"].unsqueeze(-1).float()
            return (h * msk).sum(1) / msk.sum(1)
        Y = text(tk, e5, ["query: " + x for x in titles], max_len=128)
    elif S == "S7":
        cp = f"{a.out}/captions.json"
        if not os.path.isfile(cp):
            bm = BlipForConditionalGeneration.from_pretrained(f"{M}/blip-caption-base").to(dev).eval().half(); bp = BlipProcessor.from_pretrained(f"{M}/blip-caption-base")
            class CD(Dataset):
                def __len__(self): return len(paths)
                def __getitem__(self, i): return bp(images=load(i), return_tensors="pt")["pixel_values"][0]
            caps = []
            with torch.no_grad():
                for b in DataLoader(CD(), batch_size=64, num_workers=a.workers):
                    o = bm.generate(pixel_values=b.to(dev).half(), max_new_tokens=30); caps += bp.batch_decode(o, skip_special_tokens=True)
            json.dump(caps, open(cp, "w")); del bm; torch.cuda.empty_cache()
        caps = json.load(open(cp))
        src = f"{M}/clip-vit-base-patch16"; mdl = CLIPModel.from_pretrained(src).to(dev).eval().half(); pr = CLIPProcessor.from_pretrained(src)
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b))
        Y = text(pr.tokenizer, clip_txt(mdl), caps)
    elif S == "S8":
        src = f"{M}/siglip-base-patch16-224"; mdl = AutoModel.from_pretrained(src).to(dev).eval().half(); pr = AutoProcessor.from_pretrained(src)
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b), view=thumb)
        Y = text(pr.tokenizer, lambda t: mdl.get_text_features(input_ids=t["input_ids"]), [half(t) for t in titles], max_len=64, pad="max_length")
    elif S == "S9":
        src = f"{M}/metaclip-b32-400m"; mdl = CLIPModel.from_pretrained(src).to(dev).eval().half(); pr = CLIPProcessor.from_pretrained(src)
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b), view=crop50)
        Y = text(pr.tokenizer, clip_txt(mdl), titles)
    elif S == "S10":
        src = f"{M}/align-base"; mdl = AlignModel.from_pretrained(src).to(dev).eval(); pr = AlignProcessor.from_pretrained(src)   # fp32
        X = images(pr.image_processor, lambda b: mdl.get_image_features(pixel_values=b), dtype=torch.float32)
        Y = text(pr.tokenizer, clip_txt(mdl), [five(t) for t in titles], max_len=64)
    else:
        raise ValueError(S)
    assert not np.isnan(X).any() and not np.isnan(Y).any(), f"{S}: NaN features"
    np.savez(f"{a.out}/emb_{S}.npz", img=X.astype(np.float32), txt=Y.astype(np.float32), post=POSTS, uid=uid)
    del mdl; torch.cuda.empty_cache()
