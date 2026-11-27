#!/bin/bash
# GPU health probe. Run inside a step that holds all GPUs of the node, e.g.
#   srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:l40s:10 \
#        bash /sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg/env/gpuprobe.sh
# Prints ECC counters per GPU, then tries a CUDA matmul on each visible index.
source /sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg/env/venv/bin/activate
echo "host $(hostname) slurm_cvd=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>&1
for i in 0 1 2 3 4 5 6 7 8 9; do
  r=$(CUDA_VISIBLE_DEVICES=$i timeout 120 python -c "import torch;x=torch.randn(2048,2048,device='cuda');(x@x).sum().item();print('OK',torch.cuda.get_device_properties(0).name)" 2>&1 | tail -1)
  echo "cvd=$i $r"
done
