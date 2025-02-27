import torch
import torch.nn as nn
import torch.nn.functional as F

class WideResNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(WideResNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_out = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(dropout)
        self.batch_norm_1 = nn.BatchNorm2d(hidden_channels)
        self.batch_norm_2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv_bypass = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_bypass = nn.BatchNorm2d(in_channels)

        
    def forward(self, x0):
        x = self.conv_in(x0)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.conv_out(x)

        x0 = self.conv_bypass(x0)
        x0 = self.batch_norm_bypass(x0)
        x0 = self.relu(x0)
        
        return x + x0


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSample, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.time_mlp = MLP(128, out_channels)
             
        
    def forward(self, x, t_emb):

        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(x.shape[0], t_emb_proj.shape[1], 1, 1)

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)

        x = x + t_emb_proj   
        
        x_skip = x
        x = self.max_pool(x)
        
        return x_skip, x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSample, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x_skip, x):
        x = self.upsample(x)

        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        
        return x
    
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels * 2)
        self.fc2 = nn.Linear(in_channels * 2, out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x    


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



class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        
        self.down_stages = nn.ModuleList(
            [DownSample(1, 32),
             DownSample(32, 64),
             DownSample(64, 128)
            ])
    
        self.up_stages = nn.ModuleList(
            [UpSample(256, 128),
             UpSample(128, 64),
             UpSample(64, 32)
            ])

        self.bottleneck = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()



    def forward(self, x, t):

        t_emb = get_timestep_embedding(t, 128)

        x_skips = []
        for down_stage in self.down_stages:
            x_skip, x = down_stage(x, t_emb)

            x_skips.append(x_skip)

        x = self.bottleneck(x) 

        for up_stage in self.up_stages:
            x_skip = x_skips.pop()
            x = up_stage(x_skip, x)

        return self.conv11(x)



class RDownSample(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RDownSample, self).__init__()
        self.resnet = WideResNet(in_channels=in_channels, hidden_channels=hidden_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.time_mlp = MLP(128, out_channels)
             
        
    def forward(self, x, t_emb):

        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(x.shape[0], t_emb_proj.shape[1], 1, 1)

        x = self.resnet(x)
        x = self.conv(x)
        x = self.relu(x)

        x = x + t_emb_proj   
        
        x_skip = x
        x = self.max_pool(x)
        
        return x_skip, x

class RUpSample(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RUpSample, self).__init__()
        self.resnet = WideResNet(in_channels=in_channels, hidden_channels=hidden_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x_skip, x):
        x = self.upsample(x)

        x = torch.cat([x, x_skip], dim=1)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.relu(x)
        
        return x

class RUNet(nn.Module):

    def __init__(self):
        super(RUNet, self).__init__()
        
        self.down_stages = nn.ModuleList(
            [RDownSample(1, 64, 32),
             RDownSample(32, 64, 64),
             RDownSample(64, 128, 128)
            ])
    
        self.up_stages = nn.ModuleList(
            [RUpSample(256, 256, 128),
             RUpSample(128, 128, 64),
             RUpSample(64, 64, 32)
            ])

        self.bottleneck = nn.Sequential(
            WideResNet(in_channels=128, hidden_channels=256),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()



    def forward(self, x, t):

        t_emb = get_timestep_embedding(t, 128)

        x_skips = []
        for down_stage in self.down_stages:
            x_skip, x = down_stage(x, t_emb)

            x_skips.append(x_skip)

        x = self.bottleneck(x) 

        for up_stage in self.up_stages:
            x_skip = x_skips.pop()
            x = up_stage(x_skip, x)

        return self.conv11(x)












class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer
              
        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function
            
        return x
    

# Simple U-Net architecture
class GUNet(nn.Module):
    def __init__(self, in_channels):
        super(GUNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)  # 28x28 -> 14x14
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14x14 -> 28x28
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # Skip connection from down1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.out = nn.Conv2d(64, in_channels, 1)

    def forward(self, x):
        
        # Downsampling
        d1 = self.down1(x)
        p = self.pool(d1)
        d2 = self.down2(p)
        
        # Upsampling with skip connection
        u = self.up(d2)
        u = torch.cat([u, d1], dim=1)  # Skip connection
        u = self.up1(u)
        
        # Output
        return self.out(u)