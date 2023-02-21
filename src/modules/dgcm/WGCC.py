import torch
import torch.nn as nn
import torch.nn.functional as F

class WGCC_weight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WGCC_weight,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        self.w_qs = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.w_ks = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.w_vs = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.input_dim ** 0.5)

        self.layer_norm = nn.LayerNorm(self.output_dim*2, eps=1e-6)

    def forward(self,q,k,v,adj):
        res = self.fc1(q)       
        q = self.w_qs(res)
        k = self.w_ks(res)
        v = self.w_vs(res)

        o, attn = self.attention(q, k, v, adj)

        o = torch.cat((o,res),dim=-1)

        return o, res
        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
        batch_size, n, _ = adj.size()
        q = q.view(batch_size,n,-1)
        k = k.view(batch_size,n,-1)
        v = v.view(batch_size,n,-1)

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        attn = attn.masked_fill(adj == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output.view(batch_size*n,-1), attn