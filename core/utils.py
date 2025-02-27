import torch

def beta_linear(beta_min: float, beta_max: float, max_time_steps: int, device: str):
    return torch.linspace(beta_min, beta_max, max_time_steps, device=device)
