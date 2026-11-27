# Operations: how to work on S3DF in this sprint

All times are US Pacific (PST). **Hard stop: 05:45.** Plan backwards from it:

| By | What |
|---|---|
| 05:15 | every training job stopped; only evaluation and figures after this |
| 05:35 | final `team mr` for every result |
| 05:45 | run ends; the Slurm allocations expire around 05:40-05:50 and kill anything left |

Run `date` (local VM clock, convert with `TZ=America/Los_Angeles date`) before planning.

## Where things are

| What | Where |
|---|---|
| Dataset (read only) | `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k` — 504 Zarr v2 stores `{exp}_{run}_peaknet.NNNN.zarr` (40 images each, `images` (B,H,W) float32, `labels` (B,H,W) int64, `metadata` per-image JSON, `shared_metadata/pixel_maps`), plus `*.v3.zarr` copies to ignore. Its `CLAUDE.md` describes it. Images are 1920x1920 or 1667x1665 (Jungfrau). |
| Team root on S3DF (`$PK`) | `/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg` (every bridge session's `root_dir`) |
| Your code copy on S3DF | `$PK/work/<you>/` (made by `ops/sync_up.sh`) |
| Your outputs on S3DF | `$PK/runs/<you>/<experiment>/` (logs, checkpoints, predictions, figures) |
| Python env (shared) | `source $PK/env/venv/bin/activate` — Python 3.11, torch 2.6.0+cu124, torchvision 0.21, transformers 4.57.6, timm, accelerate, zarr 2.18 (zarr 3 cannot read these stores' metadata), numcodecs, scipy, scikit-image, matplotlib, segformer-pytorch, x-unet, vit-pytorch, segmentation-models-pytorch, pycocotools, opencv, and `peaknet` (editable, from `/sdf/home/c/cwang31/codes/peaknet`) |
| Hugging Face cache | `HF_HOME=$PK/hf_home` (pre-fetched: `nvidia/mit-b0..b3,b5`, `openmmlab/upernet-convnext-{tiny,small,base}`, `openmmlab/upernet-swin-{tiny,small}`, `facebook/convnext-*-224`, `facebook/detr-resnet-50`, `PekingU/rtdetr_r50vd`, `PekingU/rtdetr_v2_r18vd`, `hustvl/yolos-small`; see `$PK/env/prefetch.log`) |
| PeakNet code (user's repo) | `/sdf/home/c/cwang31/codes/peaknet` (github.com/carbonscott/peaknet). Read it; do not edit it in place. Copy what you change into this repo (`seg/` or `patches/`). |
| Earlier PeakNet work | `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet` (training scripts, configs, its own `CLAUDE.md`). Read only. |

## Bridges: your only way to S3DF

Every agent reaches S3DF through a `bridge` session (the `bridge-skills` skill documents the
CLI). Sessions are already running; **use only yours**:

| Agents | Bridge session |
|---|---|
| lead | `peaknet-02` |
| h1, h2 | `peaknet-03` |
| h3, h4 | `peaknet-04` |
| h5, h6 | `peaknet-05` |
| h7, h8 | `peaknet-06` |

(`peaknet-01` is the boss's. Do not touch it.)

Rules:

- Always pass `--session <yours>`: `bridge --session peaknet-03 bash '...'`. Relative paths
  resolve under `$PK`; absolute paths work anywhere.
- **Two agents share one bridge, so every bridge call must return fast** (seconds, not
  minutes). Anything long (training, evaluation, staging, a big `find`, pip installs) runs
  **in the background on the remote machine**, detached, with its output in a log:

      bridge --session peaknet-03 bash 'cd $PK_WORK && setsid nohup <command> > <log> 2>&1 < /dev/null & echo started'

  Plain `cmd &` is not enough: without `setsid nohup ... < /dev/null` the bridge call hangs
  until the command ends. Then poll with short calls (`tail -n 20 <log>`).
- Global flags go before the subcommand: `bridge --session S --timeout 60 bash '...'`.
- Keep output small (`| tail`, `| head`, `wc -l`); a huge listing floods your context.
- `$DATA_ROOT` is set inside the bridge shell (`/sdf/data/lcls/ds/prj/prjcwang31`).
- Never `ssh` to S3DF yourself (login rate limits; repeated failures lock the account). If
  your bridge is dead (`bridge --session S status` fails), restart **only yours** with
  `bridge-session start --name S -- ssh sdfiana025 'uv run ~/bridge-server --root-dir /sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg'`
  once, and if that fails, wait 5 minutes before one more try and tell the lead.
- Never ssh into hutch or DAQ nodes. Nothing in this sprint needs them.

Moving files:

- Code up: commit, then `ops/sync_up.sh <your-bridge> work/<you>` (tars `seg/ scripts/
  configs/ patches/ checks/ ops/` into `$PK/work/<you>/`).
- Text back (metrics JSON, logs): `bridge --session S read <abs path> --raw > local-file`.
- Binary back (PNG figures): `ops/fetch_bin.sh <your-bridge> <remote abs path> <local path>`.

## Compute: Slurm allocations and GPUs

The boss holds exclusive `ada` nodes as long-lived allocations (`salloc --no-shell`, job names
`pkseg-n1`..`pkseg-n4`, up to 4 nodes). **Do not `salloc`, `sbatch` or `scancel` allocations
yourself.** See what is held:

    squeue -u cwang31 -o "%.10i %.10j %.8T %.10L %R"

At launch, one node is running (`pkseg-n2` = job `47394816` on `sdfada013`); three more are
queued and may start later. When one starts, the lead probes it and hands out its GPUs.

GPU health is uneven on these nodes, so **every GPU is probed before use**, and the lead keeps
the ownership table on the board (`gpus`: job id, node, index, health, owner). Use only GPUs
the board assigns to you. Probe a whole node (takes a few minutes; run it in the background):

    srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:l40s:10 \
         bash $PK/env/gpuprobe.sh > $PK/runs/<you>/probe-<node>.log 2>&1

It prints ECC counters (`ecc.errors.uncorrected.*`: anything non-zero in the volatile column
means do not use that GPU) and a CUDA test per index. A GPU whose test prints nothing timed
out (CUDA init on these nodes can take over a minute; re-probe once with more patience
before writing it off). Known at launch on `sdfada013`: indices 2, 3 and 5 passed; 0, 1 and 4
timed out on the first probe (`$PK/env/probe-47394816.log`; re-probe them); only 6 GPUs are
visible there. Until more nodes start, GPUs are scarce: share them in time (short smoke
tests first, then the runs most likely to land a statement), and do CPU-side work (code,
data checks, figures) while you wait. A node whose CUDA init fails with `CUDA unknown error` /
`cuInit 999` for every GPU is broken (this happened on `sdfada011`, which was released).

Run a job on your GPUs (one step per job; `--overlap` lets steps share the node; pin GPUs
with `CUDA_VISIBLE_DEVICES` = the indices the board gives you; always set `--ntasks=1`,
`--cpus-per-task` and `--mem` on the step, or the step grabs the whole node):

    bridge --session S bash 'cd $PK/work/<you> && setsid nohup srun --jobid=<J> --overlap --ntasks=1 \
        --cpus-per-task=16 --mem=120G --gres=gpu:l40s:10 bash -c "source $PK/env/venv/bin/activate; \
        export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2,3; \
        torchrun --standalone --nproc_per_node=2 -m seg.train ..." \
        > $PK/runs/<you>/<exp>/train.log 2>&1 < /dev/null & echo started'

(`$PK` is not set in the bridge shell; write the path out or `export PK=...` first.) The job
step id for `result.json` is in `squeue -s -u cwang31` while it runs. Budget CPUs and memory
fairly: about 10 CPUs and 90 GB of RAM per GPU.

**Compute nodes have no internet.** Downloads (pip, Hugging Face weights) happen on
`sdfiana025`, which is where `bridge bash` runs, so run them directly in `bridge bash`
(backgrounded if slow), never inside `srun`. Inside jobs set `HF_HUB_OFFLINE=1`.

**Python is uv only.** The env is shared by everyone: before installing into it, post on
the board what and why, and always install with the constraints file so torch and zarr stay
pinned:

    uv pip install --python $PK/env/venv/bin/python -c $PK/env/constraints.txt <pkg>

(`export UV_CACHE_DIR=$PK/env/uvcache` first.) Never upgrade torch, torchvision or zarr. If
you need something incompatible, make your own venv under `$PK/env/` with `uv venv`.

**Data I/O.** The dataset lives on the shared `/sdf/data` filesystem. Node-local NVMe
(`/lscratch/cwang31/` on each compute node) may be faster for repeated epochs; a past
measurement found shared scratch faster than NVMe on one node, so measure before you stage.
If you stage, stage once per node into `/lscratch/cwang31/pkseg/<store>` (coordinate on the
board so two agents do not copy the same stores), and fall back to the shared path if a
store is missing. 94 GB total; per-image decode is about 5 MB compressed, 15 MB raw float32.

## Advice: Fable

For design decisions, debugging you are stuck on, or reviewing a plan, consult a Fable
advisor: the `Agent` tool with `model: "fable"`. Give it the concrete code, numbers and
question; it cannot see your workspace unless you paste what matters. Use it when a
second opinion could save GPU hours, not for routine steps.

## Proofs

Use the `/proof-report` skill for `proof_report` (and for any evidence page). No videos.
Annotated screenshots work best: matplotlib figures made on S3DF from real predictions,
with peaks circled (matched green, missed red, false yellow), the image id and metrics
written on the figure, plus log and `squeue` excerpts rendered as images or text.

## Etiquette

- Before starting work, claim it on the board; post results and dead ends to `notes` so
  nobody repeats a failed idea.
- Do not edit another agent's `$PK/work/<agent>` or `$PK/runs/<agent>`.
- Keep `NOTES.md` (in git) as the team's lab notebook: one short dated entry per finding.
- Do not push to GitHub; the boss publishes `main` at the end.
- Never put secrets, tokens or checkpoints in git.
