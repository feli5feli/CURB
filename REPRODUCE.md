# Reproducing the experiments

All runs use `export OMP_NUM_THREADS=1`, three seeds (`S` in 0 1 2) and the hyperparameters in `config.yaml`.

## Data preparation

Set `CURB_DATA` to a cache directory.

**Fakeddit.** Place `multimodal_only_samples/multimodal_train.tsv` (Fakeddit v2) under `$CURB_DATA/meta/v2/`.

```bash
cd $CURB_DATA && python $REPO/data/make_fakeddit_manifest.py && cd $REPO
M=$CURB_DATA/manifest/d1_manifest.parquet
python data/fetch_images.py --manifest $M --out $CURB_DATA/images
python data/encode_clip.py --manifest $M --images $CURB_DATA/images --out $CURB_DATA/embed_d1
python data/encode_pipelines.py --manifest $M --images $CURB_DATA/images --out_root $CURB_DATA/pipes
python data/encode_pipelines.py --manifest $M --images $CURB_DATA/images --out_root $CURB_DATA/pipes --switch_targets

IMG=c80-none-full,c80-none-half,c80-q30-full,c80-q30-half,c80-q60-full,c80-q60-half,c90-none-full,c90-none-half,c90-q30-full,c90-q30-half,c90-q60-full,c90-q60-half,clip-none-full,clip-none-half,clip-q30-full,clip-q30-half,clip-q60-full,clip-q60-half,thumb,webp
TXT=plain,photo,post,image,caption
P="--embed $CURB_DATA/embed_d1 --pipes_dir $CURB_DATA/pipes --img_pipes $IMG --txt_pipes $TXT --n_nodes 100 --limit 80000"
for S in 0 1 2; do
  python data/build_streams.py $P --seed $S --out $CURB_DATA/up_pipe_s$S
  python data/build_streams.py $P --seed $S --mode identity_split --m_h 10 --s_max 20 --out $CURB_DATA/up_pipe_split_s$S
  python data/build_streams.py $P --seed $S --switch_at 0.5 --switch_nodes 20 \
         --switch_img_pipes c70-none-full,c70-q60-half,c75-q45-full,c85-none-half,clip-q45-full \
         --switch_txt_pipes picture,news,reddit,shows --out $CURB_DATA/up_pipe_switch_s$S
done
```

**Weibo** (Jin et al., 2017, as distributed with EANN-KDD18).

```bash
M=$CURB_DATA/manifest/d2_weibo_manifest.parquet
python data/make_weibo_manifest.py --raw $CURB_DATA/weibo_raw --out $M
python data/fetch_images.py --manifest $M --out $CURB_DATA/d2_images
python data/encode_clip.py --manifest $M --images $CURB_DATA/d2_images --out $CURB_DATA/embed_d2 --text_col text --label_col label
python data/encode_pipelines.py --manifest $M --images $CURB_DATA/d2_images --out_root $CURB_DATA/pipes_d2 --text_col text
for S in 0 1 2; do
  python data/build_streams.py --embed $CURB_DATA/embed_d2 --pipes_dir $CURB_DATA/pipes_d2 --img_pipes $IMG --txt_pipes $TXT \
         --n_nodes 50 --node_mode kmeans --seed $S --out $CURB_DATA/up_pipe_d2_s$S
done
```

**Heterogeneous services** (encoders read from `CURB_MODELS`).

```bash
python data/encode_services.py --out $CURB_DATA/services
python data/build_services.py --emb $CURB_DATA/services --n 6  --out $CURB_DATA/services
python data/build_services.py --emb $CURB_DATA/services --n 10 --out $CURB_DATA/services
```

## Experiments

Each attacked run is paired with a clean run of the same stream and seed:

```bash
python run.py --uploads $CURB_DATA/up_pipe_s$S --attack diversified --seed $S --out runs/div_s$S
python run.py --uploads $CURB_DATA/up_pipe_s$S --attack clean       --seed $S --out runs/clean_s$S
python metrics.py runs/div_s$S runs/clean_s$S
```

The main results use `--attack {clean,direct,content_init,diversified}` and `--attack adaptive`. The other settings add
the arguments below.

| experiment | arguments |
|---|---|
| signal ablations | `--method curb_no_a+u`, `curb_no_u`, `curb_no_q`, `curb_no_c` |
| calibration ablation | `--set center=false lam_c=10` / `calibrate=false lam_c=1` / `center=false calibrate=false` |
| complementarity | `--attack diversified --diag` (score terms in `diag.npz`) |
| frozen prediction model | `--set g_freeze_online=true` |
| Sybil scaling | `--uploads up_pipe_split_s$S --attack diversified --set s_used={1,5,10,20}` |
| sparse / delayed labels | `--set label_frac={0.3,0.1}` / `label_delay={2000,5000}` |
| Weibo | `--uploads up_pipe_d2_s$S --set warm_frac=0.3` |
| pipeline upgrade | `--uploads up_pipe_switch_s$S` |
| registration poisoning | `--set warm_shape=0.3` (Spread) / `warm_shift=0.6` (Shift) |
| heterogeneous services | `--uploads services/up_six` or `services/up_ten --set warm_frac=0.5 val_frac=0` |
| adaptive configurations | `--attack adaptive --set lam_P={0.1,0.25,1,2,4}` / `pgd_steps=100` / `pgd_restarts=4` / `atk_label=true` / `atk_knowledge=reg` / `adaptive_init=registration` / `nu_cap={0.3,0.5}` |
| registration history | `--set reg_posts={2000,4000,8000}` |
