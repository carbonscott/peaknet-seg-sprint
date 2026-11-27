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
