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

from optim import LrWarmupScheduler, TopkWarmupScheduler, LambdaSWarmupScheduler

def train_lorsa(lorsa: LowRankSparseAttention, cfg: LorsaTrainConfig, activation_dataset: ActivationDataset):
    # hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.buffer_size)
    # hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
    # lorsa.initialize_parameters(b_O = hook_out[filter_mask].mean(dim=0).to(cfg.lorsa_config.dtype))

    # optimizer and scheduler
    optimizer = torch.optim.Adam(lorsa.parameters(), lr=0, betas=(0.9, 0.999), weight_decay=0)
    lr_scheduler = LrWarmupScheduler(optimizer, cfg.learning_rate, cfg.final_learning_rate, cfg.lr_warm_up_tokens, cfg.lr_cool_down_tokens, cfg.total_tokens)
    lr_scheduler.update_lr(0)
    if cfg.mode == "top_k":
        k_scheduler = TopkWarmupScheduler(lorsa, cfg.start_k, cfg.end_k, cfg.k_scheduler_name, cfg.k_warm_up_tokens, cfg.total_tokens)
        k_scheduler.update_k(0)
    elif cfg.mode == "jumprelu":
        lambda_s_scheduler = LambdaSWarmupScheduler(cfg.lambda_s_final, cfg.total_tokens)
        lambda_s_scheduler.update_lambda_s(0)

    if cfg.mode == "jumprelu":
        with torch.no_grad():
            nn.init.xavier_uniform_(lorsa.W_Q)
            nn.init.xavier_uniform_(lorsa.W_K)
            
            n = lorsa.cfg.d_model
            bound_O = 1.0 / math.sqrt(n)
            nn.init.uniform_(lorsa.W_O, -bound_O, bound_O)
            
            m = lorsa.cfg.d_qk_head
            bound_V = 1.0 / math.sqrt(m)
            nn.init.uniform_(lorsa.W_V, -bound_V, bound_V)
            
            hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
            hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, _, z_pre_activation = lorsa.forward(hook_in)  # Shape: (batch_size, query_pos, n_heads, d_head)
            
            z_values = z_pre_activation.squeeze(-1)
            z_filtered = z_values[filter_mask]
            
            medians = torch.median(z_filtered, dim=0).values  # Shape: (n_heads,)
            
            current_threshold = torch.exp(lorsa.log_threshold)  # Shape: (n_heads,)
            
            lorsa.b_V.data = (current_threshold - medians).unsqueeze(-1)  # Shape: (n_heads, d_head)
            
            lorsa.initialize_parameters(b_O = hook_out[filter_mask].mean(dim=0).to(cfg.lorsa_config.dtype))
        
    elif cfg.init_scale_parameters:
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
                hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
                if cfg.mode == "top_k":
                    out, _ = lorsa.forward(hook_in)
                elif cfg.mode == "jumprelu":
                    out, _, _ = lorsa.forward(hook_in)
                else:
                    raise NotImplementedError(f"Mode {cfg.mode} not implemented")
                
                scale = torch.mean(torch.norm(hook_out[filter_mask]-lorsa.b_O, p=2, dim=-1)) / torch.mean(torch.norm(out[filter_mask]-lorsa.b_O, p=2, dim=-1))
                if cfg.init_qk_with_orig_qk == False:
                    lorsa.scale_parameters('W_Q', scale=scale)
                    lorsa.scale_parameters('b_Q', scale=scale)
                    lorsa.scale_parameters('W_K', scale=scale)
                    lorsa.scale_parameters('b_K', scale=scale)
                lorsa.scale_parameters('W_V', scale=scale)
                lorsa.scale_parameters('b_V', scale=scale)
    
    # train loop
    pbar = tqdm(total=cfg.total_tokens, desc="Training Progress", unit="tokens", dynamic_ncols=True)
    sampled_tokens = 0
    step = 0
    if cfg.log_to_wandb:
        wandb.watch(lorsa)
        if cfg.mode == "top_k":
            head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
            tokens_count = 0
        elif cfg.mode == "jumprelu":
            head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
            tokens_count = 0
            
    while sampled_tokens < cfg.total_tokens:
        # schedular update
        lr_scheduler.update_lr(sampled_tokens)
        if cfg.mode == "top_k":
            k_scheduler.update_k(sampled_tokens)
        elif cfg.mode == "jumprelu":
            lambda_s_scheduler.update_lambda_s(sampled_tokens)
        
        # get act
        hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
        hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
        
        # forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if cfg.mode == "top_k":
                out, l1_weighted = lorsa.forward(hook_in)
                mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
                loss = mse_loss
            elif cfg.mode == "jumprelu":
                out, l1_weighted, z_pre_activation = lorsa.forward(hook_in)
                mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
                
                sparsity_loss = torch.tanh(cfg.sparsity_c * l1_weighted)
                sparsity_loss = sparsity_loss[filter_mask].sum(dim=-1).mean()
                
                W_O_norms = torch.norm(lorsa.W_O, p=2, dim=2).squeeze(-1)  # (n_heads,)
                threshold = torch.exp(lorsa.log_threshold)  # (n_heads,)
                penalty_term = F.relu(threshold.unsqueeze(0).unsqueeze(0) - z_pre_activation.squeeze(-1)) * W_O_norms.unsqueeze(0).unsqueeze(0)
                penalty_loss = penalty_term[filter_mask].sum(dim=-1).mean()
                
                # 总损失
                current_lambda_s = lambda_s_scheduler.get_lambda_s()
                loss = mse_loss + current_lambda_s * sparsity_loss + cfg.lambda_p * penalty_loss
            else:
                raise NotImplementedError
        
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lorsa.parameters(),
            max_norm=cfg.clip_grad_norm if cfg.clip_grad_norm > 0 else math.inf,
        )
        optimizer.step()
        
        # update head info
        sampled_tokens += filter_mask.sum().item()
        if cfg.log_to_wandb:
            tokens_count += filter_mask.sum().item()
            if cfg.mode == "top_k":
                if cfg.lorsa_config.d_ov_head == 1:
                    top_k_mask = (l1_weighted != 0.).to(torch.int32)
                else:
                    raise NotImplementedError
                counts = torch.sum(top_k_mask[filter_mask], dim=0)
                head_use_count += counts
            elif cfg.mode == "jumprelu":
                # For jumprelu, count activations above small threshold
                if cfg.lorsa_config.d_ov_head == 1:
                    activation_mask = (l1_weighted != 0).to(torch.int32)
                else:
                    raise NotImplementedError
                counts = torch.sum(activation_mask[filter_mask], dim=0)
                head_use_count += counts
        step += 1
        
        # calculate explained variance
        with torch.no_grad():
            per_token_l2_loss = (
                (out[filter_mask] - hook_out[filter_mask]).pow(2).sum(dim=-1)
            )
            total_variance = (
                (hook_out[filter_mask] - hook_out[filter_mask].mean(0)).pow(2).sum(dim=-1)
            )
            explained_variance = 1 - per_token_l2_loss / total_variance

        # update tqdm bar
        pbar.update(filter_mask.sum().item()) 
        postfix_dict = {
            "mse_loss": round(mse_loss.item(), 3), 
            "ev": round(explained_variance.mean().item(), 2), 
            **({'k': lorsa.cfg.top_k} if cfg.mode == 'top_k' else {}),
            **({"l0": round((l1_weighted != 0.).to(torch.float32)[filter_mask].sum(dim=-1).mean().item(), 1)} if cfg.mode == "jumprelu" else {}),
        }
        if cfg.mode == "jumprelu":
            postfix_dict.update({
                "sparsity": round((current_lambda_s * sparsity_loss).item(), 4),
                "penalty": round((cfg.lambda_p * penalty_loss).item(), 4),
                "total": round(loss.item(), 3),
                "λ_s": round(current_lambda_s, 4)
            })
        pbar.set_postfix(postfix_dict)
        pbar.refresh()

        # log to wandb
        if cfg.log_to_wandb and step % cfg.log_frequency == 0:
            W_V_norms = lorsa.W_V.data.view(lorsa.cfg.n_ov_heads, lorsa.cfg.d_model).norm(dim=1)
            W_O_norms = lorsa.W_O.data.view(lorsa.cfg.n_ov_heads, lorsa.cfg.d_model).norm(dim=1)
            mean_norm_W_V = W_V_norms.mean().item()
            mean_norm_W_O = W_O_norms.mean().item()

            log_dict = {"details/sampled_tokens": sampled_tokens,
                        "details/learning_rate": lr_scheduler.get_lr(),
                        "loss/mse_loss": mse_loss.item(),
                        **({"loss/sparsity_loss": (current_lambda_s * sparsity_loss).item(),
                            "loss/penalty_loss": (cfg.lambda_p * penalty_loss).item(),
                            "loss/total_loss": loss.item(),
                            "details/lambda_s": current_lambda_s} if cfg.mode == "jumprelu" else {}),
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/l0": (l1_weighted != 0.).to(torch.float32)[filter_mask].sum(dim=-1).mean().item(),
                        "metrics/ground_truth_norm": torch.mean(torch.norm(hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/reconstructed_norm": torch.mean(torch.norm(out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/error_norm": torch.mean(torch.norm(out[filter_mask] - hook_out[filter_mask], p=2, dim=-1)).item(),
                        "metrics/b_O_norm": torch.norm(lorsa.b_O.data, p=2, dim=-1).item(),
                        "metrics/grad_norm": grad_norm.item(),
                        "metrics/W_V_norm_Mean": mean_norm_W_V,
                        "metrics/W_O_norm_Mean": mean_norm_W_O,
                        **({"sparsity/below 1e-5": (head_use_count / tokens_count < 1e-5).sum().item()} if (cfg.mode == "top_k" or cfg.mode == "jumprelu") and tokens_count > 1e5 else {}),
                        **({"sparsity/below 1e-6": (head_use_count / tokens_count < 1e-6).sum().item()} if (cfg.mode == "top_k" or cfg.mode == "jumprelu") and tokens_count > 1e6 else {}),
                        **({"sparsity/top_k": lorsa.cfg.top_k} if cfg.mode == "top_k" else {}),
                        **({"metrics/l1": l1_weighted[filter_mask].sum(dim=-1).mean().item()} if cfg.mode == "top_k" or cfg.mode == "jumprelu" else {}),
                        **({"metrics/threshold_mean": torch.exp(lorsa.log_threshold).mean().item()} if cfg.mode == "jumprelu" else {}),
                        }
            wandb.log(log_dict,
                      step=step)
            if (cfg.mode == 'top_k' or cfg.mode == 'jumprelu') and tokens_count > 1e6:
                head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
                tokens_count = 0
                
    pbar.close()
    return lorsa