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

## lucid_segformer (h5, crew C-lucid), 2026-11-27 PST

- 01:40 lucidrains `segformer-pytorch` (0.0.6) predicts at 1/4 resolution; peaks are 1-2 px. Asking it for 16 classes
  and applying `nn.PixelShuffle(4)` (+ a tiny zero-init residual conv that also sees the normalised image) gives
  full-res logits at no real cost. Trained from scratch, 7.7M params, ~4 steps/s at 16x512^2 crops on a shared L40S.
  Full-res (1920^2) val/eval needs tiling (1024) when the GPU is shared: stage-1 attention softmax alone is ~3 GB.
- 01:55 DEAD END / FIX: saturated pixels. lseg_a had val precision 0.10: 54 of 55 predicted components on a val image
  were 1-px-tall runs of 65535 along panel edges (prob 1.0). Labels never mark them (0 of 2637 peak px saturated in
  20 sampled stores). Zeroing pixels >= 60000 in the input adapter fixed it (lseg_b val P .75-.84).
- 03:00 DEAD END: lowering BCE pos_weight 10 -> 2 (lseg_c) did not cure the precision cap (best thr pinned at .9995,
  P ~.6). The remaining FPs were real-looking spots (~200 ADU over ~90 bg) the labeller left out.
- 03:20 FIX: per-image standardisation (log1p, then (x-mean)/std per image or per tile) throws away absolute
  brightness, which the labels depend on. Fine-tuning lseg_b with pointwise `seg.infer.norm_log` (lseg_d) raised val
  peak F1 .809 -> .873 within 1000 steps and to .889 by step 5000 (P .85, R .93); thresholds stop pinning at the top.
- 03:30 lseg_b whole test (thr 0.7 from val): peak F1 0.8255, pixel IoU 0.410 (landed). lseg_d (round 2) whole test (thr .998 from val): peak F1 0.8966 (P .866 R .929), pixel IoU 0.467; with flip TTA (thr .98 from val): peak F1 0.8998, pixel IoU 0.490.

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

## lucid_unet (h6, crew C-lucid), 2026-11-27 PST

- 01:25 lucidrains x-unet 0.5.0 runs every 2D conv as `nn.Conv3d` with a 1-frame kernel and builds its attention
  with einsum (full N x N matrix). Wrapper glue, same weights and maths: einsum attention -> `F.scaled_dot_product_attention`
  (max |diff| 6e-6) so full 1920x1920 inference fits, and 1-frame Conv3d -> `F.conv2d` (~1.7x more crops/s in training).
- 02:09 Dead end: `focal_dice` with the default `w_main: 1` (alpha .25) on ~1e-5 positive pixels collapsed to all
  background (xu_c: val peak F1 0.000 at step 1000, loss flat at ~0.98). A 1-crop CPU overfit test learned fine, so
  the wrapper was not at fault; `w_main: 10` (as h7's smp runs) fixed it (xu_d val F1 .75 at step 1500, .89 at 2938).
- 01:45 Sharing one L40S between two training runs: bs 8 at 512^2 plus full-res validation reached 35 GB and OOMed the
  neighbour. Use bs <= 6, `val_tile: 1024`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (x-unet then ~18 GB).
- 03:48 xu_d whole test (1529 imgs, thr .95 from 100 val imgs, tile 1024, no TTA): peak F1 0.8876 (P .897 R .878),
  pixel IoU .519. x-unet inference is slow (~1.5 s/img on a shared GPU, 38 min for the test set): budget evals early.

## detection, leaderboard, proof_report, seg/viz.py (h8, crew D-integrate), 2026-11-27 PST

- 01:17 `seg/viz.py` peak matching (8-connected centroids, max-cardinality bipartite matching on a KD-tree within
  3 px) gives the same TP count as a brute-force Hungarian match; it is the matcher used for the independent re-eval.
- 01:23 All bridges run as the same S3DF user `cwang31`: never `pkill -f <pattern>` (it can hit other agents' jobs);
  kill by exact PID only. `ops/sync_up.sh` exits silently (code 2) if `configs/` or `patches/` is missing locally.
- 02:16-02:40 Detection (HF RT-DETRv2-r18 on mask-derived boxes padded 3 px, 256-px crops upscaled 2x, tiled
  inference): val AP50 0.584 at it 400, then collapsed (0.073 @800, 0.016 @1200) with flat training losses at lr 1e-4.
  Val-selected it-400 checkpoint: whole-test mAP50 0.583 (P .671 R .520). Many FPs are real peaks whose predicted box
  is smaller than the padded label box (IoU < 0.5). A lower-LR + EMA retry was stopped to free the GPU (not run).
- 04:22 Independent re-scoring of best_model final1's saved test probability maps (zarr labels + seg.viz matcher,
  no seg/evaluate.py code) reproduces h7's whole-test numbers exactly: peak F1 0.918242, TP/FP/FN 34991/3982/2249.

## best_model + shared trainer (h7, crew D-integrate), 2026-11-27 PST

- 01:50 `seg/train.py` defaults to watch: `loss.type: focal_dice` with `w_main: 1` and focal alpha .25 can collapse to
  all-background on ~1e-5 positive pixels (h6 hit this). `w_main: 10` + dice works (smp_unet config).
- 01:45 smp Unet-ResNet34 (ImageNet, full-res decoder, pointwise `norm_log`): val peak F1 0.90 after 1000 steps (4 min).
  Full-resolution decoders are what matters for 1-2 px peaks.
- 02:05 False positives are confident (median max prob 0.99998) 1-2 px spots, none within 6 px of a labeled peak; on
  test overlays they look like real, unlabeled peaks. Precision is capped near 0.9 by the labels. Min-size / closing
  post-processing did not help (`seg/postproc.py`); raising the threshold did. Dead end: post-processing.
- 02:30 Flip TTA: +0.002 val F1. Saturated-pixel masking: no effect for smp models (`norm_log` already zeroes >= 60000);
  it did fix precision for lucid_segformer (h5). Ensembling r34 + UNet++ snapshots: +0.000 on val (shared errors).
- 03:30 Val peak F1 swings +-0.03 between checkpoints at high LR, so the val-picked checkpoint matters more than the
  architecture. UNet++-r34 best (step 5000) val .929; r34 best .917; h1's Unet-effb4 best .924.
- 04:21 best_model final1: `seg.evaluate_ensemble --subsets` (4 ckpts predicted once per val image with flip TTA, 9 subsets
  x 21 thresholds on 300 val images, one process) chose r34 last + UNet++ best at thr .70; whole test peak F1 0.9182
  (P .898 R .940). History: 0.9061 (r34 snapshot) -> 0.9141 (UNet++ alone) -> 0.9182.
- Eval cost on the shared L40S: ~0.2 s/img single model, ~0.8 s/img for 2 members x 4 flips; val selection with many
  subsets is CPU-bound (scoring), ~4-5 s/img with 4 procs.
- 04:32 Diverse ensemble member smp Unet + EfficientNet-b4 (smp-hub ImageNet weights; h7's recipe: focal x10 +
  dice, crop 512, 16 crops/step, `norm_log`): 85 min on one L40S (18.7k steps, 3-5 steps/s, 12 GB). Val peak F1
  .924 at step 10000. A 300-image val grid picked flip-TTA at thr .9995 (F1 .927; no-TTA .923 at .9999;
  saturated masking no gain because `norm_log` already zeroes >= 60000). Whole test: peak F1 0.9217 (P .896,
  R .949), pixel IoU .495, the best single model of the run; used as `best_model`.
- 04:40 Dead end: mean-prob ensemble of smp effb4 + smp UNet++ r34 scored val peak F1 .928 on the same 300 val
  images vs .927 for effb4 alone (flip-TTA, thr sweep to .9999). No test pass; `best_model` stays effb4 alone.

## evalkit + peaknet_baseline (h2, crew A-core), 2026-11-27 PST

- 01:20 `seg/evaluate.py` peak matching: centroids of 8-connected components, candidate pairs within 3 px (cKDTree),
  then one-to-one maximum bipartite matching (Hopcroft-Karp; equals Hungarian with 0/1 costs, i.e. maximises TP).
  `ndimage.center_of_mass` was the bottleneck (2 s per 1920x1920 image); a `bincount` mean of pixel coordinates gives
  identical centroids in 0.1 s.
- 02:00 PeakNet's precision was capped by saturated 65535 pixels: the wrapper only clamped them, so they became the
  brightest input and the model fired on the unlabelled runs along panel edges (pk_a val peak F1 .57, P .40 R .97).
  Zeroing pixels >= 60000 in the adapter and fine-tuning (pk_b) gave val .82; test peak F1 .805 at a val-picked thr .999.
- 03:10 For one-step train+eval jobs, `seg.evaluate` reads `eval: {val_thr: [...], val_every: k}` from the model
  config at eval start: it sweeps those thresholds on val and reports test at the val-best one. The config can be
  edited on S3DF while the job is still training, without a restart.
- 04:05 Pointwise input normalisation (`seg.infer.norm_log`, fixed centre/scale) instead of per-image standardisation
  (h5's finding) also helps PeakNet: fine-tuning pk_b with it gave val peak F1 .877 after 1000 steps (pk_b .820), and
  the best threshold moved from .998 down to .5. Dead end: a lower-lr, focal alpha .5 continuation (pk_c) did not help.
- 04:51 pk_d (pointwise-norm fine-tune of pk_b, 50 min) whole test: peak F1 .895 (P .894 R .896), pixel IoU .307 at
  a val-picked thr .99, up from pk_b's .805. This is the landed peaknet_baseline.
- 04:58 Round 2 upn_ft (fine-tune of upn_a best.pt; saturated px zeroed in the adapter, focal alpha .5, lr 5e-5,
  pos_frac .5): its step-1000 checkpoint + flip-TTA at thr .9999 won on the same 400 val images (.898 vs .8925), and gave
  whole-test peak F1 0.8912 (P .881, R .901). The fine-tune drifted to over-confidence again after step 1000.
  Not tried for lack of time: the pointwise log normalisation that helped PeakNet and lucid models (h2, h5).



## hf_segformer (h3, crew B-hf), 2026-11-27 PST

- 01:30 Peaks are 1-2 px with 4-35 per image. HF SegFormer (`nvidia/mit-b2`) wrapper: per-image robust norm
  ((x - median) / 1.4826 MAD, then asinh), learnable 1x1 conv 1->3 ch (init = replicate), pad to /32, stock 1-class
  head at 1/4 res upsampled bilinearly. Encoder LR = 0.25x head LR (`param_groups` hook for seg.train).
- 01:35 `CUDNN_STATUS_NOT_INITIALIZED` on every GPU was the shared venv (cu13 wheels over torch's cu12 cuDNN; h1
  fixed it). Disabling cuDNN (`scripts/run_nocudnn.py`) also worked, but is no longer needed.
- 02:05 Dead end: a PixelShuffle(4) full-res head on SegFormer (which helped UperNet) did worse at step 1500
  (val F1 .66, P .50 even at thr .99). FP diagnosis (`scripts/fp_diag.py`, b2 step 1000, 8 val imgs): 0/55 false
  peaks lie within 6 px of a true peak (they are not fragments); 20/55 are 1-px blobs; raw z-score at FPs is
  ~2.6-6.7, i.e. faint unlabeled spots. Precision comes from the threshold, not post-processing.
- 03:33 With focal(alpha .25)+dice, the in-run val F1 (thr capped at .9) swung .68-.84; the true optimum is
  thr .999. Val grid on 400 imgs: best.pt (step 4000) .857 plain, .867 with flip-TTA (no gain from saturated
  masking); last.pt (step 10001) only .812. A mit-b3 + focal(alpha .75)x10 + dice run (bs 8) peaked at .854
  (thr .98, step 3000) and was stopped to free the GPU for the test eval.
