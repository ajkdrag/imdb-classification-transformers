import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8) -> None:
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_keys = nn.Linear(k, k*heads, bias=False)
        self.to_queries = nn.Linear(k, k*heads, bias=False)
        self.to_values = nn.Linear(k, k*heads, bias=False)

        self.unify_heads = nn.Linear(k*heads, k)
    
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_queries(x).view(b, t, h, k)
        values = self.to_queries(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b*h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b*h, t, k)
        values = values.transpose(1, 2).contiguous().view(b*h, t, k)

        queries = queries / (k ** 0.25)
        keys = keys / (k ** 0.25)

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        return self.unify_heads(out)