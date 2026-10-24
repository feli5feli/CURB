"""Post index and CLIP ViT-B/32 text embeddings of a manifest, restricted to posts whose image was fetched.

data/build_streams.py takes the post order and labels from the index, and clusters the text embeddings into identities
for Weibo (--node_mode kmeans). The contributions themselves come from data/encode_pipelines.py.
"""
import argparse, os, numpy as np, pandas as pd, torch
from transformers import CLIPModel, CLIPProcessor

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--images", required=True)
p.add_argument("--out", required=True)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--text_col", default="clean_title")
p.add_argument("--label_col", default="2_way_label")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
df = pd.read_parquet(a.manifest)
# shard by fname, matching data/fetch_images.py
paths = [os.path.join(a.images, f[:2], f) for f in df["fname"]]
have = np.array([os.path.exists(q) and os.path.getsize(q) > 512 for q in paths])
df = df[have].reset_index(drop=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half()
tok = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer

texts = [str(t)[:300] for t in df[a.text_col]]
Y = np.zeros((len(df), 512), np.float32)
with torch.no_grad():
    for s in range(0, len(texts), a.batch):
        tk = tok(texts[s:s + a.batch], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
        f = model.text_projection(model.text_model(**tk).pooler_output)
        Y[s:s + len(f)] = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

np.save(os.path.join(a.out, "txt.npy"), Y)
df["2_way_label"] = df[a.label_col]
keep = ["uid", "created_utc"] + [c for c in ("subreddit", "author") if c in df.columns] + ["2_way_label"]
df[keep].to_parquet(os.path.join(a.out, "index.parquet"), index=False)
