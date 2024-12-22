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

from model.attention import LowRankSparseAttention
from config import LorsaConfig, LorsaTrainConfig
from activations import MultiKeyDataset, ActivationDataset, PresaveActivationDataset

from utils import LrWarmupScheduler, TopkWarmupScheduler

def train_lorsa(lorsa: LowRankSparseAttention, model: HookedTransformer, cfg: LorsaTrainConfig, activation_dataset: ActivationDataset):
    hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=8 * cfg.batch_size)
    hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
    lorsa.initialize_parameters(b_O = hook_out[filter_mask].mean(dim=0).to(cfg.lorsa_config.dtype))

    # optimizer and scheduler
    optimizer = torch.optim.Adam(lorsa.parameters(), lr=0)
    lr_scheduler = LrWarmupScheduler(optimizer, cfg.learning_rate, cfg.final_learning_rate, cfg.lr_warm_up_tokens, cfg.lr_cool_down_tokens, cfg.total_tokens)
    k_scheduler = TopkWarmupScheduler(lorsa, cfg.start_k, cfg.end_k, cfg.k_scheduler_name, cfg.k_warm_up_tokens, cfg.total_tokens)
    lr_scheduler.update_lr(0)
    k_scheduler.update_k(0)

    if cfg.init_scale_parameters:
        with torch.no_grad():
            hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
            hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
            if cfg.mode == "top_k":
                out, _ = lorsa.forward_top_k(hook_in)
            elif cfg.mode == "l2":
                out, _ = lorsa.forward_l2(hook_in)
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
        k_scheduler.update_k(sampled_tokens)
        
        # get act
        hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
        hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
        if cfg.mode == "top_k":
            out, top_k_mask = lorsa.forward_top_k(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss
        elif cfg.mode == "l2":
            out, l2 = lorsa.forward_l2(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss + cfg.l2_coef * l2[filter_mask].sum(dim=-1).mean()
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

        # update cotrol info
        sampled_tokens += filter_mask.sum().item()
        if cfg.log_to_wandb and cfg.mode == "top_k":
            tokens_count += filter_mask.sum().item()
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
        pbar.set_postfix({"mse_loss": mse_loss.item(), "ev": round(explained_variance.mean().item(), 2), 'k': lorsa.cfg.top_k})
        pbar.refresh()
        
        # log to wandb
        if cfg.log_to_wandb and step % cfg.log_frequency == 0:
            log_dict = {"details/sampled_tokens": sampled_tokens,
                        "details/learning_rate": lr_scheduler.get_lr(),
                        "loss/mse_loss": mse_loss.item(), 
                        **({"loss/l2_loss": l2.sum(dim=-1).mean().item()} if cfg.mode == "l2" else {}),
                        "metrics/explained_variance": explained_variance.mean().item() ,
                        "metrics/ground_truth_norm": torch.mean(torch.norm(hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/reconstructed_norm": torch.mean(torch.norm(out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/error_norm": torch.mean(torch.norm(out[filter_mask] - hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/b_O_norm": torch.norm(lorsa.b_O.data, p=2, dim=-1).item(),
                        "metrics/grad_norm": grad_norm.item(),
                        **({"sparsity/below 1e-5": (head_use_count / tokens_count < 1e-5).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e5 else {}),
                        **({"sparsity/below 1e-6": (head_use_count / tokens_count < 1e-6).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e6 else {}),
                        **({"sparsity/top_k": lorsa.cfg.top_k} if cfg.mode == "top_k" else {}),
                        **({"metrics/l2": l2.sum(dim=-1).mean().item()} if cfg.mode == "l2" else {}),
                        }
            wandb.log(log_dict,
                      step=step)
            if cfg.mode == 'top_k' and tokens_count > 1e6:
                head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
                tokens_count = 0
                
    pbar.close()
    return lorsa

def train_lorsa_without_forward(lorsa: LowRankSparseAttention, cfg: LorsaTrainConfig):
    dataset = load_from_disk(cfg.dataset_path)
    multi_dataset = MultiKeyDataset(
        dataset, 
        keys=['hook_in', 'hook_out', 'filter_mask'], 
        dtypes=[cfg.lorsa_config.dtype, cfg.lorsa_config.dtype, torch.bool]
    )
    dataloader = DataLoader(multi_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=4)
    data_iter = iter(dataloader)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(lorsa.parameters(), lr=0)
    lr_scheduler = LrWarmupScheduler(optimizer, cfg.learning_rate, cfg.final_learning_rate, cfg.lr_warm_up_tokens, cfg.lr_cool_down_tokens, cfg.total_tokens)
    k_scheduler = TopkWarmupScheduler(lorsa, cfg.start_k, cfg.end_k, cfg.k_scheduler_name, cfg.k_warm_up_tokens, cfg.total_tokens)
    lr_scheduler.update_lr(0)
    k_scheduler.update_k(0)
    
    if cfg.init_scale_parameters:
        with torch.no_grad():

            hook_in, hook_out, filter_mask = next(data_iter)
            hook_in, hook_out, filter_mask = hook_in.to(cfg.device), hook_out.to(cfg.device), filter_mask.to(cfg.device)

            if cfg.mode == "top_k":
                out, _ = lorsa.forward_top_k(hook_in)
            elif cfg.mode == "l2":
                out, _ = lorsa.forward_l2(hook_in)
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
        k_scheduler.update_k(sampled_tokens)
        
        # get act
        try:
            hook_in, hook_out, filter_mask = next(data_iter)
            hook_in, hook_out, filter_mask = hook_in.to(cfg.device), hook_out.to(cfg.device), filter_mask.to(cfg.device)
        except StopIteration:
            data_iter = iter(dataloader)
            hook_in, hook_out, filter_mask = next(data_iter)
            hook_in, hook_out, filter_mask = hook_in.to(cfg.device), hook_out.to(cfg.device), filter_mask.to(cfg.device)

        if cfg.mode == "top_k":
            out, top_k_indices = lorsa.forward_top_k(hook_in)

            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
        
            loss = mse_loss
        elif cfg.mode == "l2":
            out, l2 = lorsa.forward_l2(hook_in)
            
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            
            loss = mse_loss + cfg.l2_coef * l2[filter_mask].sum(dim=-1).mean()
        elif cfg.mode == "default":
            out = lorsa.forward(hook_in)
            
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            
            loss = mse_loss
        
        # back propagation and optimization
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        # update cotrol info
        sampled_tokens += filter_mask.sum().item()
        if cfg.log_to_wandb and cfg.mode == "top_k":
            tokens_count += filter_mask.sum().item()
            counts = torch.bincount(top_k_indices.reshape(-1), minlength=cfg.lorsa_config.n_ov_heads)
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
        pbar.set_postfix({"mse_loss": mse_loss.item(), "ev": round(explained_variance.mean().item(), 2), 'k': lorsa.cfg.top_k})
        pbar.refresh()
        
        # log to wandb
        if cfg.log_to_wandb and step % cfg.log_frequency == 0:
            log_dict = {"details/sampled_tokens": sampled_tokens,
                        "details/learning_rate": lr_scheduler.get_lr(),
                        "loss/mse_loss": mse_loss.item(), 
                        **({"loss/l2_loss": l2.sum(dim=-1).mean().item()} if cfg.mode == "l2" else {}),
                        "metrics/explained_variance": explained_variance.mean().item() ,
                        "metrics/ground_truth_norm": torch.mean(torch.norm(hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/reconstructed_norm": torch.mean(torch.norm(out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/error_norm": torch.mean(torch.norm(out[filter_mask] - hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/b_O_norm": torch.norm(lorsa.b_O.data, p=2, dim=-1).item(),
                        **({"sparsity/below 1e-5": (head_use_count / tokens_count < 1e-5).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e5 else {}),
                        **({"sparsity/below 1e-6": (head_use_count / tokens_count < 1e-6).sum().item()} if cfg.mode == "top_k" and tokens_count > 1e6 else {}),
                        **({"sparsity/top_k": lorsa.cfg.top_k} if cfg.mode == "top_k" else {}),
                        **({"metrics/l2": l2.sum(dim=-1).mean().item()} if cfg.mode == "l2" else {}),
                        }
            wandb.log(log_dict,
                      step=step)
            if cfg.mode == 'top_k' and tokens_count > 1e6:
                head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
                tokens_count = 0
                
    pbar.close()
    return lorsa
