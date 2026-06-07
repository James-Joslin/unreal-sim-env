from dataclasses import dataclass

@dataclass
class PPOConfig:
    # --- Learning Rate Tuning ---
    lr: float = 1.2e-4              # Adjusted for multi-head scale properties
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.08        
    vf_clip_range: float = 10.0      
    
    # --- Exploration Adjustments ---
    entropy_coef: float = 0.025      
    entropy_coef_final: float = 0.002 
    value_coef: float = 0.35         # Lowered to insulate shared encoder from value updates
    
    # --- CRITICAL GRADIENT SAFETY ---
    max_grad_norm: float = 0.3       # RESTORED: Hard ceiling against attention gradient spikes
    
    # --- Batching & Update Loops ---
    num_steps: int = 196             
    mini_batch_size: int = 512       
    update_epochs: int = 3           
    target_kl: float = 0.012         
    
    # --- Horizons & Safety Gates ---
    total_timesteps: int = 9_000_000 
    eval_interval: int = 20_000      
    save_interval: int = 100_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42
    normalize_obs: bool = True
    normalize_returns: bool = True
    
    # --- Reversion Engine Settings ---
    revert_on_regression: bool = True
    revert_patience: int = 4         # Tracked properly inside eval loop now
    revert_min_drop: float = 0.15