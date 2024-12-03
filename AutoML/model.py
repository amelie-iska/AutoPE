# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask
        x_masked = x * mask
        sum_x = torch.sum(x_masked, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
        average = sum_x / sum_mask
        return average

class Model(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, output_dim, self_attention_layers=False, repr_layers=33, dropout=0.1):
        super(Model, self).__init__()
        self.self_attention_layers = self_attention_layers
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        if self.self_attention_layers:
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.masked_avg_pool = MaskedAveragePooling()
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.repr_layers = repr_layers

    def forward(self, tokens, mask):
        results = self.pretrained_model(tokens, repr_layers=[self.repr_layers], return_contacts=True)
        token_representations = results["representations"][self.repr_layers]
        if self.self_attention_layers:
            x = token_representations.permute(1, 0, 2)
            x_skip, _ = self.self_attention(x, x, x)
            x = x + x_skip
            x = x.permute(1, 0, 2)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = token_representations
        avg_pooled = self.masked_avg_pool(x, mask)
        output = self.fc(avg_pooled)
        output = output.squeeze(-1)  # 确保输出是一维的
        return output, results