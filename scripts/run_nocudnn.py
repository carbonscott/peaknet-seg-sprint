"""Run a module with cuDNN disabled (sdfada013 GPUs fail with CUDNN_STATUS_NOT_INITIALIZED).
usage: python scripts/run_nocudnn.py seg.train --model ... (all args passed through)"""
import faulthandler
import os
import runpy
import sys

import torch

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.getcwd())
faulthandler.dump_traceback_later(600, repeat=True)
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
