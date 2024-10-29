# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-350M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 1024
gradient_accumulation_steps = 5 * 8
concat_embeddings = True

# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 100
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1
