# Lab notebook

One short dated entry per finding (PST).

## 2026-11-27 01:20 h1: split + shared loader

- Split (`results/split/result.json`, by run): train 15989 / val 1463 / test 1529 images.
  test = mfxl1025422_r0309 + mfxl1027522_r0028, val = mfxl1025422_r0310. The 25 `mfx13016_0036.*` stores are left out
  (no `_peaknet` in the name). Labels are {0,1}, peak = 1, extremely sparse (~1e-6..1e-5 of pixels).
- Loader: `from seg.data import load_split, PeakDataset, pad_collate`. `PeakDataset(stores, crop=512, pos_frac=0.9)`
  for training (crop placed on a random peak pixel with prob pos_frac); `PeakDataset(stores)` gives full native images
  (1920x1920 or 1667x1665) for eval; batch those with `pad_collate` (zero-pad to a multiple of 32, crop back with meta H, W).
  Item: image float32 (1,H,W) raw values (can be negative), label int64 (H,W), meta {store,index,identifier,H,W,y0,x0}.
  `seg.data.normalize_image` is an optional shared default (clip at 0, log1p, /8).
- Read cost on the bridge host: ~0.07 s image + ~0.11 s label (int64) per item, single thread. Use DataLoader
  `num_workers` 8-10 per GPU.
- Staging on node NVMe (optional): from `$PK/work/<you>`, run
  `srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=8 --mem=8G bash scripts/stage_lscratch.sh all > <log> 2>&1`
  once per node (claim it in notes topic infra first). It copies to `/lscratch/cwang31/pkseg/<store>` and marks each copy
  with `.complete`; `seg/data.py` uses a staged copy only when that marker exists and otherwise reads `/sdf/data`.
  Measure with `python -m seg.data --stores <store> --n 40` before relying on it.

## hf_upernet (h4, crew B-hf), 2026-11-27 PST

- 01:20 UperNet-ConvNeXt (HF `UperNetForSemanticSegmentation`) has BatchNorm after a 1x1 adaptive pool in its
  pyramid-pooling module: training with batch 1 crashes. Fix in the wrapper: keep those PPM BN layers in eval mode.
- 01:40 Stock UperNet upsamples 1/4-res logits bilinearly; peaks are 1-2 px. Swapping the 1-class classifier for a
  16-ch 1x1 conv + PixelShuffle(4) (learnable full-res logits, same cost) plus prior bias -4.6 trained well
  (upn_a val peak F1 .85 at step 1000). Stem stride 2 (decoder at 1/2 res) works but is 3.7x slower; not trained.
- 03:36 With 10*focal(alpha .75)+dice, longer training made UperNet over-confident: in-run val F1 at the capped
  thr 0.9 swung .76-.88, and the end-of-schedule last.pt reached only .864 even at thr .9999 (recall .98), while the
  step-3000 checkpoint gave .888 (thr .995) / .892 with flip-TTA (thr .5). Select checkpoints with a threshold sweep
  to >= .999 (scripts/val_grid.py) rather than the trainer's capped sweep. Saturated-pixel masking had no effect.
- 03:50 Whole test, upn_a best.pt + flip-TTA, thr .5 from val: peak F1 0.8886 (P .863, R .916), pixel IoU .494.
  Eval cost: 4-flip full-res inference ~0.6 s/img on a shared L40S (1529 test imgs in 15.6 min).

## infra + A-core (h1), 2026-11-27 PST

- 01:30 `CUDNN_STATUS_NOT_INITIALIZED` at the first conv on every GPU was the shared venv, not the hardware:
  CUDA-13 wheels (`cuda_toolkit` 13.2 via `cuda_bindings` 13.5) had overwritten torch 2.6+cu124's cu12 libraries in
  `nvidia/cudnn`, `nvidia/nccl` and `nvidia/cusparselt` (same install paths). Fixed by reinstalling torch's pins
  (`nvidia-cudnn-cu12==9.1.0.70`, `nvidia-nccl-cu12==2.21.5`, `nvidia-cusparselt-cu12==0.6.2`) with the constraints
  file. Check: `torch.backends.cudnn.version()` must print 90100. Never install anything that pulls cu13 wheels.
- 01:20 The `sdfada013` allocation has 72 CPUs and 288 GB for 6 GPUs: a step asking `--mem=90G` waits PENDING with
  an empty log once memory is claimed. Use at most 10 CPUs / 40 GB per GPU step.
- 01:45 `ops/sync_up.sh` does not copy `results/`; `seg.data.load_split` falls back to the durable manifest.
- 02:05 Dead end: PeakNet with stem stride 2 plus upstream's learned ConvTranspose upsampler (pk_p2) gave val peak
  F1 0.075 at step 1000 (precision 0.04): fragmented speckle false positives. Stopped; the stride-4 PeakNet (h2)
  is the baseline.
- 02:00 onward: every focal x10 + dice model picked the TOP of its threshold sweep (0.9, later 0.998) with recall
  far above precision. Sweep val thresholds to 0.9999 before testing (`scripts/val_grid.py` does this in one pass,
  together with flip-TTA and saturated-pixel masking).
