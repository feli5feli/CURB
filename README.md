# CURB

Code for **Many Puppets, One Hand: Cross-Modal Behavioral Evidence for Robust Aggregation in Collaborative
Misinformation Detection**.

CURB performs reliability-weighted aggregation of image–text contributions in collaborative misinformation detection.
It combines cross-modal behavioral evidence across posts and identities to defend against coordinated Sybil
manipulation.

## Setup

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1
```

Experiments use NumPy and run on CPU. Data preparation requires PyTorch, Transformers, Pillow and GPU-based encoding.

## Data

- **Fakeddit**: an 80,000-post subset constructed from the Fakeddit v2 multimodal training data.
- **Weibo**: the Weibo rumor dataset distributed with EANN-KDD18.
- **Heterogeneous services**: pretrained image and text encoders read from `CURB_MODELS`.

Set `CURB_DATA` to the local data-cache directory. Preprocessing and stream construction are described in
[REPRODUCE.md](REPRODUCE.md).

## Run

For a Fakeddit stream prepared with seed 0:

```bash
python run.py --uploads $CURB_DATA/up_pipe_s0 --attack diversified --seed 0 --out runs/div_s0
python run.py --uploads $CURB_DATA/up_pipe_s0 --attack clean --seed 0 --out runs/clean_s0
python metrics.py runs/div_s0 runs/clean_s0
```

Attacks are `clean`, `direct`, `content_init`, `diversified` and `adaptive`; `--set key=value` overrides
`config.yaml`. Accuracy and flip rate (FR) are evaluated on the second half of the test stream; ramp flip rate (RFR)
and Sybil weight share (SWS) during attack escalation.

## Reproducing the experiments

See [REPRODUCE.md](REPRODUCE.md) for data preparation, stream construction, ablations and the other evaluations.
Default hyperparameters are in `config.yaml`.
