import torch

def get_timestep_embedding(timesteps, embed_dim):
    """
    timesteps: Tensor of timestep values (e.g., [500, 200, 50])
    embed_dim: Size of the embedding (e.g., 128)
    Returns: [batch_size, embed_dim]
    """
    assert embed_dim % 2 == 0  # Must be even for sin/cos split
    half_dim = embed_dim // 2

    # Scale for frequency
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(timesteps.device)
    # Compute sin/cos embeddings
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb
