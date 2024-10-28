"""
Hyperparameter sweep configuration for W&B
"""

def get_valid_n_embd_values(n_head_values):
    """Generate n_embd values that are divisible by all possible n_head values"""
    n_embd_values = []
    base_values = [256, 384, 512, 768, 1024]
    
    # For each base value, round up to nearest number divisible by all n_head options
    for base in base_values:
        # Find LCM (Least Common Multiple) of all n_head values
        import math
        from functools import reduce
        lcm = reduce(lambda x, y: abs(x * y) // math.gcd(x, y), n_head_values)
        # Round up to nearest multiple of lcm
        adjusted = ((base + lcm - 1) // lcm) * lcm
        n_embd_values.append(adjusted)
    
    return sorted(list(set(n_embd_values)))  # Remove duplicates and sort

n_head_options = [4, 6, 8, 12]
n_embd_options = get_valid_n_embd_values(n_head_options)

sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    'metric': {
        'name': 'val/loss',
        'goal': 'minimize'   
    },
    'parameters': {
        # Sweep parameters
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.001
        },
        'n_layer': {
            'values': [4, 6, 8, 12]
        },
        'n_embd': {
            'values': n_embd_options  # These values are guaranteed to be divisible by all n_head options
        },
        'n_head': {
            'values': n_head_options
        },
        
        # Fixed parameters from train_shakespeare_char.py
        'out_dir': {'value': 'out-shakespeare'},
        'eval_interval': {'value': 100},
        'eval_iters': {'value': 200},
        'log_interval': {'value': 10},
        'always_save_checkpoint': {'value': False},
        'wandb_project': {'value': 'shakespeare-char'},
        'wandb_run_name': {'value': 'mini-gpt'},
        'dataset': {'value': 'shakespeare'},
        'gradient_accumulation_steps': {'value': 1},
        'block_size': {'value': 256},
        'dropout': {'value': 0.2},
        'max_iters': {'value': 5000},
        'lr_decay_iters': {'value': 5000},
        'min_lr': {'value': 1e-4},
        'beta2': {'value': 0.99},
        'warmup_iters': {'value': 100},
        'use_old_model': {'value': False},
        'concat_embeddings': {'value': True}
    }
}
