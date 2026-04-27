#!/bin/bash

python -u ./runner/inference.py \
input_json_path='./examples/test.json' \
sample_diffusion.N_sample=1 \
seeds=\[0,1,2\] \
dump_dir='./predictions' \
use_deepspeed_evo_attention=false
