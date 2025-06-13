import os
from typing import Dict, Optional, Tuple, Union
from jaxtyping import Float, Int
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

import einops

from config import LorsaConfig
from .kernel import _flash_attn_forward, _flash_attn_backward

class LowRankSparseAttention(nn.Module):
    def __init__(self, config: LorsaConfig):
        super(LowRankSparseAttention, self).__init__()
        
        self.cfg = config
        assert self.cfg.d_ov_head == 1, "d_ov_head must be 1 for lorsa"
        
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
    
    def set_requires_grad(self, param_name, requires_grad=True):
        for name, param in self.named_parameters():
            if name == param_name:
                param.requires_grad = requires_grad
                print(f"Set requires_grad={requires_grad} for {name}")
                return
        raise ValueError(f"Parameter '{param_name}' not found in model.")

    @torch.no_grad()
    def scale_parameters(self, param_name, scale: float):
        for name, param in self.named_parameters():
            if name == param_name:
                param.data *= scale
                print(f"Set {name} to {scale} times")
                return
        raise ValueError(f"Parameter '{param_name}' not found in model.")
    
    def cal_q_k(self, resid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.linear(resid,
                    einops.rearrange(self.W_Q, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_Q, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_Q.shape[0], self.b_Q.shape[1])
        k = F.linear(resid, 
                    einops.rearrange(self.W_K, "head_index d_model d_head -> (head_index d_head) d_model"), 
                    einops.rearrange(self.b_K, "head_index d_head -> (head_index d_head)")).reshape(resid.shape[0], resid.shape[1], self.b_K.shape[0], self.b_K.shape[1])
        
        if self.cfg.positional_embedding_type == "rotary":
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)
        
        if self.cfg.virtual_kv_num > 0:
            k = torch.cat((k, self.virtual_k.unsqueeze(0).expand(resid.shape[0], -1, -1, -1)), dim=1)
        
        return q, k # Shape: (batch_size, q_pos, n_qk_heads, d_head) (batch_size, k_pos, n_qk_heads, d_head)
    
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
        
    def cal_attn_scores(self, resid: torch.Tensor) -> torch.Tensor:
        q, k = self.cal_q_k(resid)
        
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
    
    def cal_pattern(self, resid: torch.Tensor = None, q: torch.Tensor = None, k: torch.Tensor = None) -> torch.Tensor:
        if q is not None and k is not None:
            q_ = einops.rearrange(
                q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
            )
            k_ = einops.rearrange(
                k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
            )
            attn_scores = q_ @ k_ / self.cfg.attn_scale
            
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(attn_scores)
        elif resid is not None:
            attn_scores = self.cal_attn_scores(resid)
        else:
            raise ValueError("Either 'resid' or both 'q' and 'k' must be provided, but got None for both")
        
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        
        return pattern # Shape: (batch_size, n_qk_heads, query_pos, key_pos)
    
    def cal_z_with_h(
        self, 
        resid: torch.Tensor
    ) -> Float[torch.Tensor, "batch_size query_pos n_heads d_head"]:
        """
        Get Z pattern (summing over key positions).
        """
        q, k, v = self.cal_q_k_v(resid)
        
        pattern = self.cal_pattern(q=q, k=k)
        
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        ) # Shape: (batch_size, n_ov_heads, key_pos, d_head)
        
        v_reshaped = v_.view(v_.shape[0], self.cfg.n_qk_heads, self.cfg.n_ov_heads // self.cfg.n_qk_heads, v_.shape[2], v_.shape[3])
        
        z = torch.einsum('bnqk,bnrkh->bnrqh', pattern, v_reshaped) # Shape: (batch_size, n_qk_heads, n_ov_heads/n_qk_heads, query_pos, d_head)
        
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3], z.shape[4]) # Shape: (batch_size, n_ov_heads, query_pos, d_head)
        
        z = z.permute(0, 2, 1, 3) # Shape: (batch_size, query_pos, n_ov_heads, d_head)
        
        return z
    
    def decode_z_with_W_O(self, z):
        # There may be some accuracy differences compared to using F.linear to operate directly with W_O and b_O  
        return torch.einsum("bqhd,hdm->bqm", z, self.W_O)  # Shape: (batch_size, query_pos, d_model)
    
    class FlashLorsaTopkFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, topk, W_O, use_z_relu, causal=True, softmax_scale=None):
            if _flash_attn_forward is None:
                raise ImportError("Flash attention kernel is not available.")
            
            q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
            o, lse, ctx.softmax_scale = _flash_attn_forward(
                q, k, v, causal=causal, softmax_scale=softmax_scale
            )
            
            with torch.no_grad():
                if use_z_relu:
                    l1 = torch.nn.functional.relu(o.squeeze(-1)).to(W_O.dtype) * torch.norm(W_O, p=2, dim=2).view(1, 1, W_O.shape[0])
                else:
                    l1 = torch.abs(o.squeeze(-1)).to(W_O.dtype) * torch.norm(W_O, p=2, dim=2).view(1, 1, W_O.shape[0])
                
                _, topk_indices = l1.topk(k=topk, dim=2)
                
                B, S, _ = l1.shape
                mask = torch.zeros_like(l1, dtype=bool)
                mask[torch.arange(B).view(B, 1, 1).expand(B, S, topk), 
                     torch.arange(S).view(1, S, 1).expand(B, S, topk), 
                     topk_indices] = True

            o = o * mask.unsqueeze(-1)
            l1 = l1 * mask
            
            ctx.save_for_backward(q, k, v, o, lse, topk_indices)
            ctx.causal = causal
            ctx.softmax_scale = softmax_scale
            
            return o, l1

        @staticmethod
        def backward(ctx, do, dl1):
            if _flash_attn_backward is None:
                raise ImportError("Flash attention kernel is not available. Please check your kernel module.")
                
            q, k, v, o, lse, topk_indices = ctx.saved_tensors
            
            # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
            # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
            with torch.inference_mode():
                dq = torch.zeros_like(q)
                dk = torch.zeros_like(k)
                dv = torch.zeros_like(v)
                _flash_attn_backward(
                    do,
                    q,
                    k,
                    v,
                    o,
                    lse,
                    topk_indices,
                    dq,
                    dk,
                    dv,
                    causal=ctx.causal,
                    softmax_scale=ctx.softmax_scale,
                )
            return dq, dk, dv, None, None, None, None, None
    
    def cal_out(self, resid: torch.Tensor):
        if self.cfg.mode == "top_k":
            # topk with flash lorsa
            if (self.cfg.use_flash_lorsa and 
                self.cfg.n_qk_heads == self.cfg.n_ov_heads and 
                self.cfg.virtual_kv_num == 0 and # same query and key length
                self.cfg.d_ov_head == 1):
                q, k, v = self.cal_q_k_v(resid)
                flash_dtype=torch.bfloat16
                z, l1 = self.FlashLorsaTopkFunc.apply(
                    q.to(flash_dtype), k.to(flash_dtype), v.to(flash_dtype),
                    self.cfg.top_k, self.W_O, self.cfg.use_z_relu,
                    True, 1 / self.cfg.attn_scale,
                )
                z = z.to(self.cfg.dtype)
            else:
                # flash attention
                if (self.cfg.use_flash_lorsa and 
                    self.cfg.d_qk_head * self.cfg.n_qk_heads == self.cfg.d_ov_head * self.cfg.n_ov_heads and 
                    self.cfg.virtual_kv_num == 0 and # same query and key length
                    self.cfg.d_ov_head == 1):
                    q, k, v = self.cal_q_k_v(resid)
                    v = v.reshape(v.shape[0], v.shape[1], self.cfg.n_qk_heads, self.cfg.d_qk_head)
                    flash_dtype=torch.bfloat16
                    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                        z = F.scaled_dot_product_attention(q.transpose(1, 2).to(flash_dtype), k.transpose(1, 2).to(flash_dtype), v.transpose(1, 2).to(flash_dtype))
                    z = z.transpose(1, 2)
                    z = z.reshape(z.shape[0], z.shape[1], self.cfg.n_ov_heads, self.cfg.d_ov_head)
                    
                    with torch.no_grad():
                        # l1: (batch_size, query_pos, n_heads)
                        if self.cfg.use_z_relu:
                            l1 = torch.nn.functional.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads).to(flash_dtype)
                        else:
                            l1 = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
                        
                        # topk_values, topk_indices = l1.topk(k=self.cfg.top_k, dim=2)
                        
                        # B, S, _ = l1.shape
                        # mask = torch.zeros_like(l1, dtype=bool)
                        # mask[torch.arange(B).view(B, 1, 1).expand(B, S, self.cfg.top_k), torch.arange(S).view(1, S, 1).expand(B, S, self.cfg.top_k), topk_indices] = True
                
                        # topk
                        k_smallest = self.cfg.n_ov_heads - self.cfg.top_k + 1

                        # top_k_values: (batch_size, query_pos)
                        top_k_values, _ = torch.kthvalue(l1, k=k_smallest, dim=2)
                        
                        # top_k_mask: (batch_size, query_pos, n_heads)
                        mask = l1 > top_k_values.unsqueeze(-1)
                        
                    z = z.to(self.cfg.dtype)
                    
                # simple attention
                else:
                    z = self.cal_z_with_h(resid)
                    
                    with torch.no_grad():
                        # l1: (batch_size, query_pos, n_heads)
                        if self.cfg.use_z_relu:
                            l1 = torch.nn.functional.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
                        else:
                            l1 = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
                        
                        _, topk_indices = l1.topk(k=self.cfg.top_k, dim=2)
                        
                        B, S, _ = l1.shape
                        mask = torch.zeros_like(l1, dtype=bool)
                        mask[torch.arange(B).view(B, 1, 1).expand(B, S, self.cfg.top_k), torch.arange(S).view(1, S, 1).expand(B, S, self.cfg.top_k), topk_indices] = True
                
                        # topk
                        # k_smallest = self.cfg.n_ov_heads - self.cfg.top_k + 1

                        # # top_k_values: (batch_size, query_pos)
                        # top_k_values, _ = torch.kthvalue(l1, k=k_smallest, dim=2)
                        
                        # # top_k_mask: (batch_size, query_pos, n_heads)
                        # mask = l1 > top_k_values.unsqueeze(-1)
                    
                z = z * mask.unsqueeze(-1)
                l1 = l1 * mask
                    
        elif self.cfg.mode == "jumprelu":
            # flash attention
            if self.cfg.use_flash_lorsa and self.cfg.d_qk_head * self.cfg.n_qk_heads == self.cfg.d_ov_head * self.cfg.n_ov_heads and self.cfg.d_ov_head == 1:
                q, k, v = self.cal_q_k_v(resid)
                v = v.reshape(resid.shape[0], resid.shape[1], self.cfg.n_qk_heads, self.cfg.d_qk_head)
                z = F.scaled_dot_product_attention(q, k, v, attention_mask=self.mask[None, None, -q.shape[1]:, -k.shape[1]:])
            # simple attention
            else:
                z = self.cal_z_with_h(resid)
                
            if self.cfg.use_z_relu:
                l1 = torch.nn.functional.relu(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)
            else:
                l1 = torch.abs(z.squeeze(-1)) * torch.norm(self.W_O, p=2, dim=2).view(1, 1, self.cfg.n_ov_heads)

        # out: (batch_size, query_pos, d_model)
        out = self.decode_z_with_W_O(z)

        return out, l1

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        out, l1 = self.cal_out(resid) # Shape: (batch_size, query_pos, d_model), (batch_size, query_pos, n_heads)
        out = out + self.b_O
        return out, l1 # Shape: (batch_size, query_pos, d_model), (batch_size, query_pos, n_heads)
    
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
        hook_out = hook_out * math.sqrt(self.cfg.d_model) / self.cfg.avg_norm['out'] #TODO! 这里的hook_out经过了scale_norm,尺度和SAE的重构就不一样了，需要重新搞一下MSE
        return hook_in, hook_out
    
    @torch.no_grad()
    def fold_W_O_into_W_V(self):
        O_norm = torch.norm(self.W_O, p=2, dim=2)  # n_ov_head d_ov_head
        self.W_O /= O_norm[:, :, None]
        self.W_V *= O_norm[:, None, :]
        self.b_V *= O_norm

        return self
    
    @torch.no_grad()
    def rescale_parameters_for_expected_average_only_in(self):
        input_scale_factor = torch.tensor(self.cfg.d_model, dtype=torch.float, device=self.cfg.device).sqrt() / self.cfg.avg_norm['in']
        output_scale_factor = torch.tensor(self.cfg.d_model, dtype=torch.float, device=self.cfg.device).sqrt() / self.cfg.avg_norm['out']

        self.W_V.data *= input_scale_factor

        self.W_O.data /= output_scale_factor
        self.b_O.data /= output_scale_factor                                                                                                    

        self.W_Q.data *= input_scale_factor
        
        self.W_K.data *= input_scale_factor

        return self
    
    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None, dtype = None):
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
        lorsa.rescale_parameters_for_expected_average_only_in()

        if dtype is not None:
            lorsa.cfg.dtype=dtype
            lorsa.to(dtype)
            torch.cuda.empty_cache()

        return lorsa

    def cal_per_key_position_z_with_h(
        self, 
        v: torch.Tensor, 
        pattern: torch.Tensor,
        interested_head_mask: Float[torch.Tensor, "reduced_n_ov_heads"] | None = None,
    ) -> Float[torch.Tensor, "batch_size query_pos key_pos n_heads d_head"]:
        """
        Get Z pattern of each key pos without summing, might consume a lot of memory.
        `interested_head_mask` indexes the heads we are interested.
        """

        if interested_head_mask is None:
            interested_head_mask = torch.arange(v.size(2), dtype=torch.int32, device=v.device)
                
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        )[:, interested_head_mask, :, :] # Shape: (batch_size, reduced_n_ov_heads, key_pos, d_head)
        
        lorsa_rate = self.cfg.n_ov_heads // self.cfg.n_qk_heads
        pattern_ = pattern[:, interested_head_mask // lorsa_rate, :, :] # Shape: (batch_size, reduced_n_ov_heads, query_pos, key_pos)
        
        # Shape: (batch_size, reduced_n_ov_heads, query_pos, key_pos, d_head)
        z = pattern_[:, :, :, :, None] * v_[:, :, None, :, :] 

        # Rearrange z to the desired shape
        z = einops.rearrange(
            z, "batch head_index query_pos key_pos d_head -> batch query_pos key_pos head_index d_head"
        ) # shape: (batch_size, query_pos, key_pos, reduced_n_ov_heads, d_head)
        
        return z