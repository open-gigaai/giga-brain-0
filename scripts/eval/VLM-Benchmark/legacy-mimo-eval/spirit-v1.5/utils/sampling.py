# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import torch

def sample_beta(alpha: float, beta: float, bsize: int, device) -> torch.Tensor:
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    return m.sample((bsize,)).to(device).reshape((bsize,))

def sample_noise(shape, device) -> torch.Tensor:
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

def sample_time(bsize: int, device) -> torch.Tensor:
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)
