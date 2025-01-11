import os
import math

import sys

import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components import Attention
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import copy

import wandb

from models.lorsa import LowRankSparseAttention
from config import LorsaConfig, LorsaTrainConfig
from activations import MultiKeyDataset, ActivationDataset, PresaveActivationDataset

from optim import LrWarmupScheduler, TopkWarmupScheduler

def train_lorsa(lorsa: LowRankSparseAttention, cfg: LorsaTrainConfig, activation_dataset: ActivationDataset):
    hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.buffer_size)
    hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
    lorsa.initialize_parameters(b_O = hook_out[filter_mask].mean(dim=0).to(cfg.lorsa_config.dtype))

    # optimizer and scheduler
    optimizer = torch.optim.Adam(lorsa.parameters(), lr=0)
    lr_scheduler = LrWarmupScheduler(optimizer, cfg.learning_rate, cfg.final_learning_rate, cfg.lr_warm_up_tokens, cfg.lr_cool_down_tokens, cfg.total_tokens)
    lr_scheduler.update_lr(0)
    if cfg.mode == "top_k":
        k_scheduler = TopkWarmupScheduler(lorsa, cfg.start_k, cfg.end_k, cfg.k_scheduler_name, cfg.k_warm_up_tokens, cfg.total_tokens)
        k_scheduler.update_k(0)

    if cfg.init_scale_parameters:
        with torch.no_grad():
            hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
            hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
            if cfg.mode == "top_k":
                out, _, _ = lorsa.forward_top_k(hook_in)
            elif cfg.mode == "l1":
                out, _ = lorsa.forward_l1(hook_in)
            elif cfg.mode == "default":
                out = lorsa.forward(hook_in)
            
            lorsa.scale_parameters(scale=torch.mean(torch.norm(hook_out[filter_mask]-lorsa.b_O, p=2, dim=-1)) / torch.mean(torch.norm(out[filter_mask]-lorsa.b_O, p=2, dim=-1)))
    
    # train loop
    pbar = tqdm(total=cfg.total_tokens, desc="Training Progress", unit="tokens", dynamic_ncols=True)
    sampled_tokens = 0
    step = 0
    if cfg.log_to_wandb:
        wandb.watch(lorsa)
        if cfg.mode == "top_k":
            head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
            tokens_count = 0
            
    while sampled_tokens < cfg.total_tokens:
        # schedular update
        lr_scheduler.update_lr(sampled_tokens)
        if cfg.mode == "top_k":
            k_scheduler.update_k(sampled_tokens)
        
        # get act
        hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
        hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
        if cfg.mode == "top_k":
            out, top_k_z, l1 = lorsa.forward_top_k(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss
        elif cfg.mode == "l1":
            out, l1 = lorsa.forward_l1(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss + cfg.l1_coef * l1[filter_mask].sum(dim=-1).mean()
        elif cfg.mode == "default":
            out = lorsa.forward(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss
        
        # back propagation and optimization
        optimizer.zero_grad() 
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lorsa.parameters(),
            max_norm=cfg.clip_grad_norm if cfg.clip_grad_norm > 0 else math.inf,
        )
        optimizer.step() 

        # update head info
        sampled_tokens += filter_mask.sum().item()
        if cfg.log_to_wandb and cfg.mode == "top_k":
            tokens_count += filter_mask.sum().item()
            if cfg.lorsa_config.d_ov_head == 1:
                top_k_mask = (top_k_z.squeeze(dim=-1) != 0.).to(torch.int32)
            else:
                raise NotImplementedError
            counts = torch.sum(top_k_mask[filter_mask], dim=0)
            head_use_count += counts
        step += 1
        
        # calculate explained variance
        with torch.no_grad():
            per_token_l2_loss = (
                (out[filter_mask] - hook_out[filter_mask]).pow(2).sum(dim=-1)
            )
            total_variance = (
                (out[filter_mask] - out[filter_mask].mean(0)).pow(2).sum(dim=-1)
            )
            explained_variance = 1 - per_token_l2_loss / total_variance

        # update tqdm bar
        pbar.update(filter_mask.sum().item()) 
        pbar.set_postfix({
            "mse_loss": round(mse_loss.item(), 3), 
            "ev": round(explained_variance.mean().item(), 2), 
            **({'k': lorsa.cfg.top_k} if cfg.mode == 'top_k' else {}),
            **({"l1": round(l1[filter_mask].sum(dim=-1).mean().item(), 1)} if cfg.mode == "top_k" or cfg.mode == "l1" else {}),
        })
        pbar.refresh()

        # log to wandb
        if cfg.log_to_wandb and step % cfg.log_frequency == 0:
            W_V_norms = lorsa.W_V.data.view(lorsa.cfg.n_ov_heads, lorsa.cfg.d_model).norm(dim=1)
            W_O_norms = lorsa.W_O.data.view(lorsa.cfg.n_ov_heads, lorsa.cfg.d_model).norm(dim=1)
            mean_norm_W_V = W_V_norms.mean().item()
            q1_norm_W_V = W_V_norms.kthvalue(k=(lorsa.cfg.n_ov_heads + 3) // 4).values.item()
            median_norm_W_V = W_V_norms.median().item()
            q3_norm_W_V = W_V_norms.kthvalue(k=3 * (lorsa.cfg.n_ov_heads + 1) // 4).values.item()
            mean_norm_W_O = W_O_norms.mean().item()
            q1_norm_W_O = W_O_norms.kthvalue(k=(lorsa.cfg.n_ov_heads + 3) // 4).values.item()
            median_norm_W_O = W_O_norms.median().item()
            q3_norm_W_O = W_O_norms.kthvalue(k=3 * (lorsa.cfg.n_ov_heads + 1) // 4).values.item()

            log_dict = {"details/sampled_tokens": sampled_tokens,
                        "details/learning_rate": lr_scheduler.get_lr(),
                        "loss/mse_loss": mse_loss.item(), 
                        **({"loss/l1_loss": l1.sum(dim=-1).mean().item()} if cfg.mode == "l1" else {}),
                        "metrics/explained_variance": explained_variance.mean().item() ,
                        "metrics/ground_truth_norm": torch.mean(torch.norm(hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/reconstructed_norm": torch.mean(torch.norm(out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/error_norm": torch.mean(torch.norm(out[filter_mask] - hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/b_O_norm": torch.norm(lorsa.b_O.data, p=2, dim=-1).item(),
                        "metrics/grad_norm": grad_norm.item(),
                        "metrics/W_V_norm_Third_quartile": q3_norm_W_V,
                        "metrics/W_V_norm_Median": median_norm_W_V,
                        "metrics/W_V_norm_First_quartile": q1_norm_W_V,
                        "metrics/W_V_norm_Mean": mean_norm_W_V,
                        "metrics/W_O_norm_Third_quartile": q3_norm_W_O,
                        "metrics/W_O_norm_Median": median_norm_W_O,
                        "metrics/W_O_norm_First_quartile": q1_norm_W_O,
                        "metrics/W_O_norm_Mean": mean_norm_W_O,
                        **({"sparsity/below 1e-5": (head_use_count / tokens_count < 1e-5).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e5 else {}),
                        **({"sparsity/below 1e-6": (head_use_count / tokens_count < 1e-6).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e6 else {}),
                        **({"sparsity/top_k": lorsa.cfg.top_k} if cfg.mode == "top_k" else {}),
                        **({"metrics/l1": l1[filter_mask].sum(dim=-1).mean().item()} if cfg.mode == "top_k" or cfg.mode == "l1" else {}),
                        }
            wandb.log(log_dict,
                      step=step)
            if cfg.mode == 'top_k' and tokens_count > 1e6:
                head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
                tokens_count = 0
                
    pbar.close()
    return lorsa