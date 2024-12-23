from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

class toy_attn(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_heads: int):
        super(toy_attn, self).__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        
        self.attn_scale = self.d_head ** -0.5
        
        self.W_Q = nn.Parameter(
            torch.empty(
                self.n_heads,
                self.d_model,
                self.d_head,
            )
        )
        self.W_K = nn.Parameter(
            torch.empty(
                self.n_heads,
                self.d_model,
                self.d_head,
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.n_heads,
                self.d_model,
                self.d_head,
            )
        )
        self.W_O = nn.Parameter(
            torch.empty(
                self.n_heads,
                self.d_head,
                self.d_model,
            )
        )
        self.b_Q = nn.Parameter(
            torch.zeros(self.n_heads, self.d_head),
            requires_grad=False
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.n_heads, self.d_head),
            requires_grad=False
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.n_heads, self.d_head),
            requires_grad=False
        )
        self.b_O = nn.Parameter(
            torch.zeros(self.d_model),
            requires_grad=False
        )
        
        nn.init.kaiming_normal_(self.W_Q, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_K, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_V, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_O, mode='fan_in', nonlinearity='relu')

        nn.init.zeros_(self.b_Q)
        nn.init.zeros_(self.b_K)
        nn.init.zeros_(self.b_V)
        nn.init.zeros_(self.b_O)
        
    def forward(self, resid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(resid, 
                    einops.rearrange(self.W_Q, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_Q, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_Q.shape[0], self.b_Q.shape[1])
        k = F.linear(resid, 
                    einops.rearrange(self.W_K, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_K, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_K.shape[0], self.b_K.shape[1])
        v = F.linear(resid, 
                    einops.rearrange(self.W_V, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_V, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_V.shape[0], self.b_V.shape[1])
        
        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )
        attn_scores = q_ @ k_ / self.attn_scale
        import torch

        query_pos = q.shape[1]
        key_pos = k.shape[1]

        mask = torch.tril(torch.ones(query_pos, key_pos)).to(q.device)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)

        mask = mask.unsqueeze(0).unsqueeze(1)

        attn_scores = attn_scores + mask

        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern.to(v.device)
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        )
        pattern_ = pattern
        z = einops.rearrange(
            pattern_ @ v_,
            "batch head_index query_pos d_head -> batch query_pos head_index d_head",
        )
        
        z_heads = z.reshape(z.shape[0], z.shape[1], self.n_heads, self.d_head)  # Shape: (batch_size, seq_len, n_heads, d_head)

        out_heads = torch.einsum('b s h d, h d m -> b s h m', z_heads, self.W_O)  # Shape: (batch_size, seq_len, n_heads, d_model)
        
        out = out_heads.sum(dim=2)
        
        return out, out_heads, pattern
        
    
    
class toy_attn_model(nn.Module):
    def __init__(self, num_embeddings: int, num_unembeddings: int, d_model: int, d_head: int, n_heads: int):
        
        super(toy_attn_model, self).__init__()
        
        self.attn = toy_attn(d_model, d_head, n_heads)
        self.num_embeddings = num_embeddings
        self.num_unembeddings = num_unembeddings
        self.d_model = d_model
        
        self.embed = nn.Embedding(self.num_embeddings, self.d_model)
        self.unembed = nn.Linear(self.d_model, self.num_unembeddings, bias=False)
        
    def input_to_embed(self, input: torch.Tensor) -> torch.Tensor:
        return self.embed(input)

    def forward(self, input: torch.Tensor):
        resid = self.input_to_embed(input)
        
        out, _, _ = self.attn(resid)
        
        logits = self.unembed(out)
            
        return logits
    
class toy_sparse_attn(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_heads: int, use_l1_loss: bool = True, use_topk: bool = False, top_k: int = None):
        
        super(toy_sparse_attn, self).__init__()
        
        self.d_model = d_model
        
        self.attn = toy_attn(d_model, d_head, n_heads)
        
        self.use_l1_loss = use_l1_loss
        self.use_topk = use_topk
        if self.use_topk:
            self.top_k = top_k
        
    def forward(self, resid: torch.Tensor):
        if self.use_l1_loss:
            # batch_size, seq_len, d_model   batch_size, seq_len, n_heads, d_model   batch_size n_heads query_pos key_pos)
            out,                             out_heads,                              pattern = self.attn(resid)
            l1 = torch.linalg.vector_norm(out_heads, dim=-1).sum(dim=2) # batch_size, seq_len
            return out, out_heads, pattern, l1
        if self.use_topk:
            out, out_heads, pattern = self.attn(resid)
            l1 = torch.linalg.vector_norm(out_heads, dim=-1) # batch_size, seq_len, n_heads
            top_k_indices = torch.topk(l1, self.top_k, dim=2).indices # batch_size, seq_len, top_k
            top_k_out_heads = torch.gather(out_heads, dim=2, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)) # batch_size, seq_len, top_k, d_model
            top_k_out = top_k_out_heads.sum(dim=2) # batch_size, seq_len, d_model
            return out, out_heads, top_k_indices, top_k_out_heads, top_k_out, pattern