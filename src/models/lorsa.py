import os
from typing import Dict, Optional, Tuple, Union
from jaxtyping import Float, Int
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from config import LorsaConfig


class LowRankSparseAttention(nn.Module):
    def __init__(self, config: LorsaConfig):
        super(LowRankSparseAttention, self).__init__()
        
        self.cfg = config
        
        if self.cfg.attn_scale is None:
            self.cfg.attn_scale = self.cfg.d_qk_head ** 0.5
        
        self.W_Q = nn.Parameter(
            torch.empty(
                self.cfg.n_qk_heads,
                self.cfg.d_model,
                self.cfg.d_qk_head,
                dtype=self.cfg.dtype
            )
        )
        self.W_K = nn.Parameter(
            torch.empty(
                self.cfg.n_qk_heads,
                self.cfg.d_model,
                self.cfg.d_qk_head,
                dtype=self.cfg.dtype
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.cfg.n_ov_heads,
                self.cfg.d_model,
                self.cfg.d_ov_head,
                dtype=self.cfg.dtype
            )
        )
        self.W_O = nn.Parameter(
            torch.empty(
                self.cfg.n_ov_heads,
                self.cfg.d_ov_head,
                self.cfg.d_model,
                dtype=self.cfg.dtype
            )
        )
        self.b_Q = nn.Parameter(
            torch.zeros(self.cfg.n_qk_heads, self.cfg.d_qk_head, dtype=self.cfg.dtype),
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_qk_heads, self.cfg.d_qk_head, dtype=self.cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_ov_heads, self.cfg.d_ov_head, dtype=self.cfg.dtype),
        )
        self.b_O = nn.Parameter(
            torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype),
        )
        
        if self.cfg.virtual_kv_num > 0:
            self.virtual_k = nn.Parameter(
                torch.empty(
                    self.cfg.virtual_kv_num,
                    self.cfg.n_qk_heads,
                    self.cfg.d_qk_head,
                    dtype=self.cfg.dtype
                )
            )
            self.virtual_v = nn.Parameter(
                torch.zeros(
                    self.cfg.virtual_kv_num,
                    self.cfg.n_ov_heads,
                    self.cfg.d_ov_head,
                    dtype=self.cfg.dtype
                ),
                requires_grad=False
            )
        
        nn.init.kaiming_normal_(self.W_Q, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_K, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_V, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_O, mode='fan_in', nonlinearity='relu')
        
        if self.cfg.virtual_kv_num > 0:
            nn.init.kaiming_normal_(self.virtual_k, mode='fan_in', nonlinearity='relu')

        nn.init.zeros_(self.b_Q)
        nn.init.zeros_(self.b_K)
        nn.init.zeros_(self.b_V)
        nn.init.zeros_(self.b_O)
        
        causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx + self.cfg.virtual_kv_num)).bool(), diagonal=self.cfg.virtual_kv_num)
        self.register_buffer("mask", causal_mask)
        self.register_buffer("IGNORE", torch.tensor(-torch.inf))
        
        
        if self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position.
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)
    
    def initialize_parameters(self, **kwargs):
        allowed_params = {"W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"}
        
        for param_name, param_value in kwargs.items():
            if param_name in allowed_params:
                setattr(self, param_name, nn.Parameter(param_value, requires_grad=True))
            else:
                raise ValueError(f"Invalid parameter name: {param_name}. Allowed names are: {allowed_params}")

    def scale_parameters(self, scale: float):
        self.W_Q.data *= scale
        self.W_K.data *= scale
        self.W_V.data *= scale
        # self.W_O.data *= scale
    
    def cal_q_k_v(self, resid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(resid,
                    einops.rearrange(self.W_Q, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_Q, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_Q.shape[0], self.b_Q.shape[1])
        k = F.linear(resid, 
                    einops.rearrange(self.W_K, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_K, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_K.shape[0], self.b_K.shape[1])
        v = F.linear(resid, 
                    einops.rearrange(self.W_V, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_V, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_V.shape[0], self.b_V.shape[1])
        
        if self.cfg.positional_embedding_type == "rotary":
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        if self.cfg.virtual_kv_num > 0:
            k = torch.cat((k, self.virtual_k.unsqueeze(0).expand(resid.shape[0], -1, -1, -1)), dim=1)
            v = torch.cat((v, self.virtual_v.unsqueeze(0).expand(resid.shape[0], -1, -1, -1)), dim=1)
        
        return q, k, v # Shape: (batch_size, q_pos, n_qk_heads, d_head) (batch_size, k_pos, n_qk_heads, d_head) (batch_size, k_pos, n_ov_heads, d_head)
        
    def cal_attn_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )
        attn_scores = q_ @ k_ / self.cfg.attn_scale
        
        # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
        attn_scores = self.apply_causal_mask(attn_scores)
        
        return attn_scores # Shape: (batch_size, n_qk_heads, query_pos, key_pos)
    
    def cal_pattern(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        attn_scores = self.cal_attn_scores(q, k)
        
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        
        return pattern # Shape: (batch_size, n_qk_heads, query_pos, key_pos)
    
    def cal_z_with_k_h(self, v: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        ) # Shape: (batch_size, n_ov_heads, key_pos, d_head)
        
        pattern_ = pattern.repeat_interleave(int(self.cfg.n_ov_heads / self.cfg.n_qk_heads), dim=1) # Shape: (batch_size, n_ov_heads, query_pos, key_pos)
        
        z = pattern_[:, :, :, :, None] * v_[:, :, None, :, :] # Shape: (batch_size, n_heads, query_pos, key_pos, d_head)

        # Rearrange z to the desired shape
        z = einops.rearrange(
            z, "batch head_index query_pos key_pos d_head -> batch query_pos key_pos head_index d_head"
        ) # shape: (batch_size, query_pos, key_pos, n_heads, d_head)
        
        return z
    
    def cal_z_with_h(self, v: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        ) # Shape: (batch_size, n_ov_heads, key_pos, d_head)
        
        pattern_ = pattern.repeat_interleave(int(self.cfg.n_ov_heads / self.cfg.n_qk_heads), dim=1) # Shape: (batch_size, n_ov_heads, query_pos, key_pos)
        
        z = torch.matmul(pattern_, v_)  # Shape: (batch_size, n_heads, query_pos, d_head)

        # Rearrange z to the desired shape
        z = einops.rearrange(
            z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"
        ) # shape: (batch_size, query_pos, n_heads, d_head)
        
        return z
    
    def decode_z_with_W_O(z):
        # There may be some accuracy differences compared to using F.linear to operate directly with W_O and b_O  
        return torch.einsum("bqhd,hdm->bqm", z, self.W_O)  # Shape: (batch_size, query_pos, d_model)
    
    def cal_out(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each query position without b_O 
        '''
        
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)
        
        z = self.cal_z_with_h(v, pattern) # Shape: (batch_size, query_pos, n_heads, d_head)

        return self.decode_z_with_W_O(z)
    
    def cal_out_with_k(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each query position and each key position, without b_O 
        '''
        
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)
        
        z = self.cal_z_with_k_h(v, pattern) # Shape: (batch_size, query_pos, key_pos, n_heads, d_head)
                
        return self.decode_z_with_W_O(z)
        
    
    def cal_out_with_k_h(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each head, each query position, each key position, without b_O 
        '''
        
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)
        
        z = self.cal_z_with_k_h(v, pattern) # Shape: (batch_size, query_pos, key_pos, n_heads, d_head)

        return self.decode_z_with_W_O(z)
    
    def cal_out_with_h(self, resid: torch.Tensor, mode = None) -> torch.Tensor:
        
        '''
        Calculate the output of each query position and each head without b_O 
        '''
        
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)
        
        z = self.cal_z_with_h(v, pattern) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        out = torch.einsum("bqnh,nhm->bqnm", z, self.W_O) # Shape: (batch_size, query_pos, n_heads, d_model)
        
        if mode == 'top_k' or (mode is None and self.cfg.mode == 'top_k'):
            
            with torch.no_grad():
                l1 = torch.linalg.vector_norm(out, dim=-1) # batch_size, query_pos, n_heads
                top_k_indices = torch.topk(l1, self.cfg.top_k, dim=2).indices # batch_size, query_pos, top_k
                
                batch_size, seq_len, n_heads = out.shape[:3]
                head_mask = torch.zeros((batch_size, seq_len, n_heads), dtype=torch.int32).to(self.cfg.device)
                head_mask.scatter_(2, top_k_indices, 1)
                
                out = out * head_mask.unsqueeze(-1)
        
        return out # Shape: (batch_size, query_pos, n_heads, d_model)
    
    def cal_out_top_k(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of heads which have top_k l1 norm, without b_O
        '''
        
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)
        
        z = self.cal_z_with_h(v, pattern) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        out_heads = torch.einsum("bqnh,nhm->bqnm", z, self.W_O) # Shape: (batch_size, query_pos, n_heads, d_model)
        
        with torch.no_grad():
            l1 = torch.linalg.vector_norm(out_heads, dim=-1) # batch_size, query_pos, n_heads
            top_k_indices = torch.topk(l1, self.cfg.top_k, dim=2).indices # batch_size, query_pos, top_k
        
        top_k_out_heads = torch.gather(out_heads, dim=2, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.cfg.d_model)) # batch_size, seq_len, top_k, d_model
        top_k_out = top_k_out_heads.sum(dim=2) # batch_size, seq_len, d_model
        
        # batch_size, seq_len, n_heads, d_model = out_heads.shape
        # mask = torch.zeros(batch_size, seq_len, n_heads, device=out_heads.device, dtype=torch.bool)
        # mask.scatter_(2, top_k_indices, True)
        # top_k_out = (out_heads * mask.unsqueeze(-1)).sum(dim=2)  # Shape: (batch_size, seq_len, d_model)
        
        return top_k_out, top_k_indices
    
    def cal_out_top_k_for_ov1(self, resid: torch.Tensor):
        # q, k, v: (batch_size, query_pos, n_heads, d_head)
        q, k, v = self.cal_q_k_v(resid)

        # pattern: (batch_size, n_heads, query_pos, key_pos)
        pattern = self.cal_pattern(q, k)
        
        # z: (batch_size, query_pos, n_heads, d_head)
        z = self.cal_z_with_h(v, pattern)

        with torch.no_grad():
            # abs_z: (batch_size, query_pos, n_heads)
            if self.cfg.use_z_relu:
                abs_z = F.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
            else:
                abs_z = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
            
            k_smallest = self.cfg.n_ov_heads - self.cfg.top_k + 1

            # top_k_values: (batch_size, query_pos)
            top_k_values, _ = torch.kthvalue(abs_z, k=k_smallest, dim=2)
            
            # top_k_mask: (batch_size, query_pos, n_heads)
            top_k_mask = abs_z >= top_k_values.unsqueeze(-1)

        top_k_z = z * top_k_mask.unsqueeze(-1)

        # out: (batch_size, query_pos, d_model)
        out = torch.einsum("bqhd,hdm->bqm", top_k_z, self.W_O)

        return out, top_k_z

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        out = self.cal_out(resid) # Shape: (batch_size, query_pos, d_model)
        out = out + self.b_O
        return out # Shape: (batch_size, query_pos, d_model)
    
    def forward_top_k(self, resid: torch.Tensor) -> torch.Tensor:
        if self.cfg.d_ov_head == 1:
            out, top_k_z = self.cal_out_top_k_for_ov1(resid) # Shape: (batch_size, query_pos, d_model) (batch_size, seq_len, top_k)
        else:
            raise NotImplementedError('Not implemented yet')
            # out, top_k_z = self.cal_out_top_k(resid) # Shape: (batch_size, query_pos, d_model) (batch_size, seq_len, top_k)
        out = out + self.b_O
        return out, top_k_z # Shape: (batch_size, query_pos, d_model)
    
    def forward_l1(self, resid: torch.Tensor) -> torch.Tensor:
        out = self.cal_out_with_h(resid)
        l1 = torch.linalg.vector_norm(out, dim=-1) # batch_size, seq_len, n_heads
        out = out.sum(dim=2) + self.b_O
        return out, l1
    
    def forward_with_k(self, resid: torch.Tensor)-> torch.Tensor:
        out = self.cal_out_with_k(resid) # Shape: (batch_size, query_pos, key_pos, d_model)
        return out # Shape: (batch_size, query_pos, key_pos, d_model)
    
    
    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x
    
    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # x = x.repeat(1, 1, 1, self.cfg.rotary_scale)
        x = x.repeat_interleave(self.cfg.rotary_scale, dim=-1)
        
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        rotary_cos = self.rotary_cos[
            None, : x_pos, None, :
        ]
        rotary_sin = self.rotary_sin[
            None, : x_pos, None, :
        ]
        x_rotated = x_rot * rotary_cos + x_flip * rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)


    def apply_causal_mask(
        self,
        attn_scores: Float[torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"],
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        # Index back to front to ensure local attention works
        final_mask = self.mask[None, None, -query_ctx_length:, -key_ctx_length:]  # [1, 1, query_pos, key_pos]

        attn_scores = attn_scores.to(final_mask.device)
        return torch.where(final_mask, attn_scores, self.IGNORE)

    def scale_norm(
        self, 
        hook_in: Float[torch.Tensor, "batch seq_len d_model"], 
        hook_out: Float[torch.Tensor, "batch seq_len d_model"]
        ):
        hook_in = hook_in * math.sqrt(self.cfg.d_model) / self.cfg.avg_norm['in']
        hook_out = hook_out * math.sqrt(self.cfg.d_model) / self.cfg.avg_norm['out']
        return hook_in, hook_out
    
    @classmethod
    def from_pretrained(cls, path: str):
        cfg = LorsaConfig.from_pretrained(path=path)
        lorsa = cls(cfg)

        state_dict_path = os.path.join(path, 'final.pth')
        state_dict = torch.load(
            state_dict_path, 
            weights_only=True, 
            map_location=cfg.device
        )

        lorsa.load_state_dict(state_dict)
        return lorsa
