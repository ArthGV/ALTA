import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
    
class DiffusionModelVid(nn.Module):
    def __init__(self, 
                 device,
                 in_channels: int = 1,        # B&W frames
                 timestep_dim: int = 10, 
                 condition_dim: int = 12,
                 num_timesteps: int = 100, 
                 beta_start: float = 0.01, 
                 beta_end: float = 0.1,
                 ):

        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        alphas = 1. - self.betas
        self.alphas = alphas.to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.device = device

        self.time_mlp = nn.Sequential(
            nn.Linear(1, timestep_dim),
            nn.SiLU()
        )

        self.cond_proj = nn.Linear(timestep_dim + condition_dim, in_channels)

        mid_ch = 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(mid_ch, in_channels, 3, padding=1),
        )

    def predict_noise(self, z: torch.Tensor, t: torch.Tensor, a: torch.Tensor, c: torch.Tensor):
        # z: [B, C, H, W] - noisy frame
        # t: [B, 1]        - fractional timestep
        # a: [B, A]        - action class
        # c: [B, 1]        - time coherence variable

        t_embed = self.time_mlp(t)                                   # [B, timestep_dim]

        # ③ merge timestep + condition into a per-channel bias, broadcast over H,W
        cond = torch.cat([t_embed, a, c], dim=-1)                    # [B, timestep_dim + condition_dim]
        bias = self.cond_proj(cond)                                  # [B, C]
        bias = bias.unsqueeze(-1).unsqueeze(-1)                      # [B, C, 1, 1]

        z = z + bias          # inject conditioning                  # [B, C, H, W]
        z = self.enc(z)       # encode                               # [B, mid_ch, H, W]
        z = self.dec(z)       # decode back to frame shape           # [B, C, H, W]
        return z

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        # ④ view() reshape to broadcast over [B, C, H, W]
        sqrt_alphas_cumprod          = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise, noise

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=0, high=self.num_timesteps - 1, size=(n,))

    def training_step(self, x0: torch.Tensor, a: torch.Tensor, c: torch.Tensor):
        # x0: [B, C, H, W]
        t = self.sample_timesteps(x0.shape[0]).to(self.device)
        x_noisy, noise = self.forward_diffusion(x0, t)
        t_frac = t.float() / (self.num_timesteps - 1)
        predicted_noise = self.predict_noise(x_noisy, t_frac.view(-1, 1), a, c)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def ddim_sample(self, z, a, c, sample_num_timesteps=50):
        for t in reversed(range(0, sample_num_timesteps - 1)):
            batch = z.shape[0]
            t_train = int(t * (self.num_timesteps - 1) / (sample_num_timesteps - 1))
            t_frac = torch.full((batch, 1), t / (sample_num_timesteps - 1), device=self.device)

            eps = self.predict_noise(z, t_frac, a, c)
            alpha_t = self.alphas_cumprod[t_train]
            alpha_prev = self.alphas_cumprod[t_train-1]

            x0 = (z - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            z = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1 - alpha_prev) * eps
        return x0
    
    def ddim_invert(self, x0, a, c, num_timesteps=50):
        xt = x0
        for t in range(num_timesteps - 1):
            batch = xt.shape[0]
            t_train = int(t * (self.num_timesteps - 1) / (num_timesteps - 1))
            t_frac = torch.full((batch, 1), t / (num_timesteps - 1), device=self.device)

            eps = self.predict_noise(xt, t_frac, a, c)
            alpha_t = self.alphas_cumprod[t_train]
            alpha_next = self.alphas_cumprod[t_train + 1]
 
            x0_pred = (xt - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            if t>0:
                xt = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * eps
        return xt
    
def _init_diffusion_model(device, model_config_path: str):
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)
        feature_dim = model_config['dataset']['feature_dim']
        num_timesteps = model_config['model_parameters']['num_timesteps']
        timestep_dim = model_config['model_parameters']['timestep_dim']
        condition_dim = model_config['model_parameters']['condition_dim']
        beta_start = model_config['model_parameters']['beta_start']
        beta_end = model_config['model_parameters']['beta_end']

    model = DiffusionModelVid(device, 1, timestep_dim, condition_dim, num_timesteps, beta_start, beta_end)
    return model, model_config
    
def init_diffusion_model(device, model_config_path: str):
    model, model_config = _init_diffusion_model(device, model_config_path)
    model.train()
    model.to(device)
    return model, model_config

def load_diffusion_model(device, model_config_path: str, model_weights_path: str):
    model, model_config = _init_diffusion_model(device, model_config_path)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, model_config