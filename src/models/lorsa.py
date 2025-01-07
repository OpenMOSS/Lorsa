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
    
    def cal_per_key_position_z_with_h(
        self, 
        v: torch.Tensor, 
        pattern: torch.Tensor,
    ) -> Float[torch.Tensor, "batch_size query_pos key_pos n_heads d_head"]:
        """
        Get Z pattern of each key pos without summing, might consume a lot of memory.
        """
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        ) # Shape: (batch_size, n_ov_heads, key_pos, d_head)
        
        pattern_ = pattern.repeat_interleave(
            self.cfg.n_ov_heads // self.cfg.n_qk_heads, 
            dim=1,
        ) # Shape: (batch_size, n_ov_heads, query_pos, key_pos)
        
        z = pattern_[:, :, :, :, None] * v_[:, :, None, :, :] # Shape: (batch_size, n_heads, query_pos, key_pos, d_head)

        # Rearrange z to the desired shape
        z = einops.rearrange(
            z, "batch head_index query_pos key_pos d_head -> batch query_pos key_pos head_index d_head"
        ) # shape: (batch_size, query_pos, key_pos, n_heads, d_head)
        
        return z
    
    def cal_z_with_h(
        self, 
        v: torch.Tensor, # Shape: (batch_size, key_pos, n_ov_heads, d_head)
        pattern: torch.Tensor # Shape: (batch_size, n_qk_heads, query_pos, key_pos)
    ) -> Float[torch.Tensor, "batch_size query_pos n_heads d_head"]:
        """
        Get Z pattern (summing over key positions).
        """
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        ) # Shape: (batch_size, n_ov_heads, key_pos, d_head)
        
        v_reshaped = v_.view(v_.shape[0], self.cfg.n_qk_heads, self.cfg.n_ov_heads // self.cfg.n_qk_heads, v_.shape[2], v_.shape[3])
        
        z = torch.einsum('bnqk,bnrkh->bnrqh', pattern, v_reshaped) # Shape: (batch_size, n_qk_heads, n_ov_heads/n_qk_heads, query_pos, d_head)
        
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3], z.shape[4]) # Shape: (batch_size, n_ov_heads, query_pos, d_head)
        
        z = z.permute(0, 2, 1, 3) # Shape: (batch_size, query_pos, n_ov_heads, d_head)
        
        return z
    
    def cal_q_k_v_pattern(self, resid):
        q, k, v = self.cal_q_k_v(resid) # Shape: (batch_size, query_pos, n_heads, d_head)
        
        pattern = self.cal_pattern(q, k) # Shape: (batch_size, n_heads, query_pos, key_pos)

        return q, k, v, pattern
    
    def decode_z_with_W_O(self, z):
        # There may be some accuracy differences compared to using F.linear to operate directly with W_O and b_O  
        return torch.einsum("bqhd,hdm->bqm", z, self.W_O)  # Shape: (batch_size, query_pos, d_model)
    
    def cal_out(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each query position without b_O 
        '''
        
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)

        z = self.cal_z_with_h(v, pattern) # Shape: (batch_size, query_pos, n_heads, d_head)

        return self.decode_z_with_W_O(z)
    
    def cal_out_from_per_key_position(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each query position and each key position, without b_O 
        '''
        
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
        z = self.cal_per_key_position_z_with_h(v, pattern) # Shape: (batch_size, query_pos, key_pos, n_heads, d_head)
                
        return self.decode_z_with_W_O(z)
        
    
    def cal_out_from_per_key_position_h(self, resid: torch.Tensor) -> torch.Tensor:
        
        '''
        Calculate the output of each head, each query position, each key position, without b_O 
        '''
        
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
        z = self.cal_per_key_position_z_with_h(v, pattern) # Shape: (batch_size, query_pos, key_pos, n_heads, d_head)

        return self.decode_z_with_W_O(z)
    
    def cal_out_with_h(self, resid: torch.Tensor, mode = None) -> torch.Tensor:
        
        '''
        Calculate the output of each query position and each head without b_O 
        '''
        
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
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
        
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
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
    
    def cal_out_top_k_for_ov1(self, resid: torch.Tensor, filter_mask: torch.Tensor):
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
        # z: (batch_size, query_pos, n_heads, d_head)
        z = self.cal_z_with_h(v, pattern)

        with torch.no_grad():
            # l1: (batch_size, query_pos, n_heads)
            if self.cfg.use_z_relu:
                l1 = F.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
            else:
                l1 = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
            
            # topk
            k_smallest = self.cfg.n_ov_heads - self.cfg.top_k + 1

            # top_k_values: (batch_size, query_pos)
            top_k_values, _ = torch.kthvalue(l1, k=k_smallest, dim=2)
            
            # top_k_mask: (batch_size, query_pos, n_heads)
            top_k_mask = l1 >= top_k_values.unsqueeze(-1)

            '''
            # batch topk
            k_smallest = (self.cfg.n_ov_heads - self.cfg.top_k) * filter_mask.numel() + 1

            top_k_values, _ = torch.kthvalue(l1.contiguous().view(-1), k=k_smallest)

            top_k_mask = l1 >= top_k_values
            
            # approximate batch topk
            k_smallest = self.cfg.n_ov_heads - self.cfg.top_k + 1

            # top_k_values: (batch_size, query_pos)
            top_k_values, _ = torch.kthvalue(l1, k=k_smallest, dim=2)
            
            # top_k_mask: (batch_size, query_pos, n_heads)
            top_k_mask = l1 >= top_k_values[filter_mask].median()

            # sentence topk
            k_smallest = filter_mask.sum(dim=-1) * (self.cfg.n_ov_heads - self.cfg.top_k) + 1

            l1[~filter_mask] = float('-inf')

            flat_l1 = l1.contiguous().view(l1.shape[0], -1)

            top_k_values = torch.stack([flat_l1[i].kthvalue(k_smallest[i]).values for i in range(l1.shape[0])])

            top_k_mask = l1 >= top_k_values.view(-1, 1, 1)
            '''

        top_k_z = z * top_k_mask.unsqueeze(-1)

        # out: (batch_size, query_pos, d_model)
        out = self.decode_z_with_W_O(top_k_z)

        return out, top_k_z.squeeze(-1), l1 * top_k_mask

    def cal_out_l1_for_ov1(self, resid: torch.Tensor):
        q, k, v, pattern = self.cal_q_k_v_pattern(resid)
        
        # z: (batch_size, query_pos, n_heads, d_head)
        z = self.cal_z_with_h(v, pattern)

        if self.cfg.use_z_relu:
            l1 = F.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
        else:
            l1 = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)

        # out: (batch_size, query_pos, d_model)
        out = torch.einsum("bqhd,hdm->bqm", z, self.W_O)

        return out, l1

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        out = self.cal_out(resid) # Shape: (batch_size, query_pos, d_model)
        out = out + self.b_O
        return out # Shape: (batch_size, query_pos, d_model)
    
    def forward_top_k(self, resid: torch.Tensor) -> torch.Tensor:
        if self.cfg.d_ov_head == 1:
            out, top_k_z, l1 = self.cal_out_top_k_for_ov1(resid) # Shape: (batch_size, query_pos, d_model) (batch_size, seq_len, n_heads, d_head)
        else:
            raise NotImplementedError('Not implemented yet')
            # out, top_k_z = self.cal_out_top_k(resid) # Shape: (batch_size, query_pos, d_model) (batch_size, seq_len, top_k)
        out = out + self.b_O
        return out, top_k_z, l1 # Shape: (batch_size, query_pos, d_model), (batch_size, seq_len, n_heads, d_head), (batch_size, seq_len, n_heads)
    
    def forward_l1(self, resid: torch.Tensor) -> torch.Tensor:
        if self.cfg.d_ov_head == 1:
            out, l1 = self.cal_out_l1_for_ov1(resid)
        else:
            raise NotImplementedError('Not implemented yet')
            # out = self.cal_out_with_h(resid)
        out = out + self.b_O
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

    '''
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
    '''

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

        # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
        # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
        if self.cfg.use_NTK_by_parts_rope:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
            )
            factor = self.cfg.NTK_by_parts_factor
            low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
            high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
            old_context_len = self.cfg.old_context_len

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            freq = 1 / inv_freq_llama
        else:
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
    
    @torch.no_grad()
    def fold_W_O_into_W_V(self):
        O_norm = torch.norm(self.W_O, p=2, dim=2)  # n_ov_head d_ov_head
        self.W_O /= O_norm[:, :, None]
        self.W_V *= O_norm[:, None, :]
        self.b_V *= O_norm

        return self
    
    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None):
        cfg = LorsaConfig.from_pretrained(path=path)
        device = cfg.device if device is None else device
        
        lorsa = cls(cfg)
        lorsa.to(device, non_blocking=True)

        state_dict_path = os.path.join(path, 'final.pth')
        state_dict = torch.load(
            state_dict_path, 
            weights_only=True, 
            map_location=device
        )

        lorsa.load_state_dict(state_dict)
        return lorsa
