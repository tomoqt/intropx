# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-arxiver'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'arxiver'
wandb_run_name = 'mini-gpt'

dataset = 'arxiver'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256 # context of up to 256 previous characters
use_old_model = False  # If True, use model_old.py instead of model.py
concat_embeddings = True
# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2                                                                                          
compile = True
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# Add a flag to choose between old and new model versions
use_old_model = False  # Set to True to use the old model as a baseline
