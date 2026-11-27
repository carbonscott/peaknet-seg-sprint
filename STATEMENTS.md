# PeakNet segmentation sprint: statements

**Read `OPERATIONS.md` before doing anything.** It tells you how to reach S3DF, which bridge
session and GPUs are yours, how to run jobs, and the hard stop time.

## Goal

Train the best possible **Bragg-peak segmentation** models on the `peaknet10k` dataset
(X-ray diffraction images from LCLS MFX, with per-pixel peak masks). Build them by reusing
existing projects, with **surgical changes only**:

- Hugging Face (`transformers` models such as SegFormer and UperNet, `timm` backbones from
  the Hub, pretrained weights from huggingface.co).
- lucidrains on GitHub (pip packages such as `segformer-pytorch`, `x-unet`, `vit-pytorch`).
- The user's own `carbonscott/peaknet` repo (the baseline model).

Do **not** write architectures from scratch. What you write is glue: data loading, input
adapters (1-channel detector images into a 3-channel pretrained model, normalization,
padding, patching), loss functions, training loops built from upstream examples, and config.
Record every upstream project you reuse, with a version or commit, in `result.json`.

Balance exploration and exploitation. Try several model families and ideas (losses for
sparse targets, resolution and tiling, pretrained versus scratch, augmentation, ensembles,
TTA, thresholds), but get each statement **done** with a real number early, then improve it.
A merged mediocre result beats an unmerged great one at 05:45.

## How a statement is done

A statement `<name>` is done when `results/<name>/result.json` (and whatever it references)
is on `main` and `pkcheck <name>` passes in the merge job. The check is `checks/pkcheck.py`
(frozen; read it, it is the exact contract). Once `results/<name>/result.json` exists on
`main`, every later merge must keep that statement passing.

The check cannot see S3DF, so it trusts the numbers you write. **The boss audits them**:
every metric must come from a real run whose log is at the S3DF path you give. A fabricated
or copied number gets the agent removed.

Files a merge may change: `seg/`, `scripts/`, `configs/`, `patches/` (code, yaml, json, md,
sh), `results/` (json, md, html, png, csv, txt), `README.md`, `NOTES.md`. Nothing else.
No file in `results/` over 3 MB (15 MB for `.html`). **Never put checkpoints or data in git.**

## Shared protocol (everyone uses these)

- **Split:** `results/split/result.json` plus the loader `seg/data.py`. Every model trains on
  `train`, picks checkpoints and thresholds on `val`, and reports on the **whole `test` set**.
- **Metrics:** `seg/evaluate.py`. The team-wide headline metric is **test peak F1 at a
  3 px tolerance** (`eval.peak.f1`, with `eval.peak_tol_px = 3`). Pixel IoU is second.
- Predictions are evaluated at the images' native resolution against the stored labels.

## Statements

| Statement | Needs | What it is |
|---|---|---|
| `split` | nothing | the fixed train/val/test split and the shared loader |
| `evalkit` | nothing | the shared metric code |
| `peaknet_baseline` | split, evalkit | the user's PeakNet model, trained and evaluated under the shared protocol |
| `hf_segformer` | split, evalkit | Hugging Face `SegformerForSemanticSegmentation` (e.g. `nvidia/mit-b*`) |
| `hf_upernet` | split, evalkit | Hugging Face `UperNetForSemanticSegmentation` (ConvNeXt or Swin backbone) |
| `lucid_segformer` | split, evalkit | lucidrains `segformer-pytorch` |
| `lucid_unet` | split, evalkit | lucidrains `x-unet` (or another lucidrains U-Net) |
| `best_model` | some models | the best combination the team can make; must beat every other model statement on test peak F1 |
| `leaderboard` | 3+ models | `results/leaderboard/result.json` and a leaderboard table in `README.md` |
| `proof_report` | models | an HTML proof page with annotated screenshots (use the `/proof-report` skill) |
| `detection` | split | LOW PRIORITY: a peak **object detector** (boxes), labels derived from the masks |

Model statements can start before `split` and `evalkit` land (write the model and training
glue, smoke-test it on a few stores), but their reported numbers must use the merged split
and evaluator.

### split

`results/split/result.json`:

```json
{"statement": "split",
 "stores": {"train": ["mfxl1025422_r0309_peaknet.0000.zarr", "..."], "val": ["..."], "test": ["..."]},
 "n_images": {"train": 0, "val": 0, "test": 0},
 "label_values": [0, 1], "peak_label": 1,
 "remote_manifest": "/sdf/.../split.json",
 "notes": "how runs were assigned"}
```

- Use the **504 Zarr v2 stores** (`{exp}_{run}_peaknet.NNNN.zarr`, not `*.v3.zarr`) in
  `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k`. Cover at least 450.
- Split **by run**: all stores of one `{exp}_{run}` go to one set (no leakage). Aim for about
  80/10/10 by images, with at least 1000 test images, and each experiment represented where
  possible.
- `seg/data.py` is the shared PyTorch `Dataset` over a list of stores (images float32,
  labels int64, per-image metadata), usable by every model. Keep it simple and fast; note
  in `NOTES.md` how to stage stores on node-local NVMe (`/lscratch`).
- `label_values` are the label ids actually seen; `peak_label` is the id of the peak class.

### evalkit

`seg/evaluate.py` (numpy and scipy only at import time, so the check can import it) defines:

- `pixel_metrics(pred, target, cls=1) -> dict` with `tp, fp, fn, precision, recall, f1, iou`.
  Pixels equal to `cls` are positive. 0/0 counts as 1.0 for precision, recall and IoU; F1 is
  0 when precision + recall is 0.
- `peak_metrics(pred, target, cls=1, tol=3.0) -> dict` with `tp, fp, fn, precision, recall, f1`.
  Peaks are the 8-connected components of `== cls`; each is represented by its centroid. A
  predicted and a true peak match when their centroids are at most `tol` px apart, with
  **one-to-one** matching (e.g. Hungarian). Same 0/0 rules.
- Both accept a 2D `(H, W)` or a batch `(B, H, W)` and **sum counts over the batch** (micro
  average).
- Also give a command-line entry point (`python -m seg.evaluate ...` or similar) that
  evaluates a model's saved test predictions or a checkpoint on the test split and prints
  JSON. Every model's `eval.eval_cmd` is the command that ran it. Import torch lazily inside
  that entry point, not at module level.

`results/evalkit/result.json` is `{"statement": "evalkit", "notes": "..."}`.

### Model statements (peaknet_baseline, hf_segformer, hf_upernet, lucid_segformer, lucid_unet, best_model)

`results/<name>/result.json`:

```json
{"statement": "hf_segformer",
 "model": "SegFormer mit-b2, 1-ch adapter, focal+dice",
 "upstream": [{"url": "https://huggingface.co/nvidia/mit-b2", "ref": "transformers 4.57.6"}],
 "code": ["seg/models/hf_segformer.py", "configs/hf_segformer.yaml"],
 "train": {"slurm_jobid": "47394816.12", "gpus": 2, "wallclock_min": 55, "train_images": 16000,
           "config": "configs/hf_segformer.yaml", "remote_log": "/sdf/.../train.log"},
 "checkpoint": "/sdf/.../best.pt",
 "eval": {"split": "test", "n_images": 2000, "peak_tol_px": 3,
          "pixel": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
          "peak": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
          "eval_cmd": "python -m seg.evaluate ...", "remote_log": "/sdf/.../eval.log"},
 "proof": ["results/hf_segformer/overlay_test.png"],
 "notes": "what was tried, what helped, what did not"}
```

- `eval.n_images` must equal the split's test size: evaluate on the whole test set.
- `upstream` must include a Hugging Face, lucidrains or `carbonscott/peaknet` URL (any
  upstream for `best_model`).
- `proof`: at least one **annotated** PNG made from real test predictions: image, ground
  truth and prediction overlaid, with matched, missed and false peaks marked and the
  metrics written on the figure.
- `best_model` also has `"based_on": [...]` (what it combined) and must have a test peak F1
  at least as high as every other model statement on `main`. It may be an ensemble,
  a fine-tune of another, a different threshold or TTA, or anything else that is honest.
  If someone later beats it, update `best_model`.

### leaderboard

`results/leaderboard/result.json` = `{"statement": "leaderboard", "entries": [{"statement",
"model", "peak_f1", "pixel_iou", ...}]}`, sorted by `peak_f1`, best first. It must list every
model statement that has a result on `main`, with numbers equal to theirs, and `README.md`
must have the same table (mention each statement name). Keep it current as results land.

### proof_report

`results/proof_report/index.html`: a self-contained page made with the `/proof-report` skill
(no videos; annotated screenshots work best), with `results/proof_report/result.json` =
`{"statement": "proof_report", ...}`. It needs at least 5 `<img>` and must cover every model
statement that has a result on `main`. Show real evidence: annotated prediction overlays,
training curves, the leaderboard, `squeue`/log excerpts, and an independent re-evaluation of
the best checkpoint.

### detection (low priority)

The dataset has no box labels. Derive them (e.g. one box per connected peak component,
padded) and train an object detector reusing Hugging Face (e.g. DETR, RT-DETR, YOLOS,
Deformable DETR in `transformers`) or lucidrains code. `results/detection/result.json`:
`statement, model, labels_from, upstream[{url,ref}], checkpoint, eval{split: "test",
n_images, map50, precision, recall, remote_log}, proof[...]`. Work on it only when the
segmentation statements are moving and a GPU would otherwise sit idle.
