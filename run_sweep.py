import wandb
import subprocess
import sys
from config.sweep_config import sweep_config

def validate_config(config):
    """Validate that n_embd is divisible by n_head"""
    if config['n_embd'] % config['n_head'] != 0:
        raise ValueError(f"n_embd ({config['n_embd']}) must be divisible by n_head ({config['n_head']})")

def main():
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="shakespeare")
    
    def run_agent():
        # Initialize wandb run
        run = wandb.init()
        
        # Validate the configuration
        try:
            validate_config(wandb.config)
        except ValueError as e:
            print(f"Invalid configuration: {e}")
            run.finish(exit_code=1)
            return
        
        # Get the sweep parameters for this run
        params = []
        for key, value in wandb.config.items():
            params.append(f"--{key}={value}")
        
        # Command to run the training script with the config file and sweep parameters
        train_cmd = [
            sys.executable,
            "train.py",
            "config/train_shakespeare.py",
            "--wandb_log=True"
        ] + params
        
        print("Running command:", " ".join(train_cmd))  # Debug print
        subprocess.run(train_cmd)
        
    # Run the sweep
    wandb.agent(sweep_id, function=run_agent, count=10)  # Run 10 experiments

if __name__ == "__main__":
    main()
