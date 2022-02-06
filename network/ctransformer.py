import torch
import torch.nn.functional as F
from torch import nn
from network.transformer_block import TransformerBlock


class CTransformer(nn.Module):
    def __init__(self, k, heads, depth, max_seq_length, num_tokens, num_classes, device):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(max_seq_length, k)
        self.device = device

        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(k, heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.to_probs = nn.Linear(k, num_classes)

    def forward(self, x):
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        positions = torch.arange(t, device=self.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        x = self.to_probs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    ctrans = CTransformer(64, 4, 2, 10, 10000, 10)
    data = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    got = ctrans(data)
    print(got.shape)
