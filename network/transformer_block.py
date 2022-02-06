import torch.nn.functional as F
from torch import nn
from network.self_attn import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads)    

        self.norm_1 = nn.LayerNorm(k)
        self.norm_2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
    
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm_1(attended + x)

        fed_forward = self.ff(x)
        return self.norm_2(fed_forward + x)

        


