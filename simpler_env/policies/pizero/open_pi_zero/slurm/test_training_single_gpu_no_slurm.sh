#!/bin/bash

# no wandb logging
# logging GPU memory usage
# smaller global batch size
# small resource for dataloading
# try saving model

# first batch will take a while with torch.compile as model being compiled
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
    scripts/run.py \
    --config-name=bridge \
    device=cuda:0 \
    debug=True \
    wandb=null \
    log_dir=results/test/ \
    global_batch_size=32 \
    per_device_batch_size=16 \
    flow_sampling=beta \
    data.train.shuffle_buffer_size=10000 \
    data.train.num_parallel_calls=10 \
    eval_freq=50 \
    eval_size=64 \
    save_model_freq=100 \
    lora=False \
    quantize=False \
    use_amp=True \
    use_torch_compile=True \
    use_bf16=True
# 'resume_checkpoint_path="...fractal_train[:95%]_tp4_beta...ckpt....pt"'
