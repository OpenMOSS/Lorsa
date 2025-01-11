import os
import json

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from fvcore.nn import FlopCountAnalysis

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from tqdm import tqdm

from models.lorsa import LowRankSparseAttention
from models.load_tl_model import load_tl_model
from config import LorsaTrainConfig, LorsaAnalyzeConfig, DataGenConfig, LorsaEvaluateConfig, LorsaConfig
from train import train_lorsa
from activations import TextActivationDataset, PresaveActivationDataset



def evaluate_lorsa(cfg):
    if cfg.dataset_type == 'text':
        model = load_tl_model(cfg.model_name)
        model.offload_params_after(f'blocks.{cfg.layer}.hook_attn_out', torch.tensor([[0]], device=cfg.lorsa_config.device))
        model.eval()
        activation_dataset = TextActivationDataset(cfg=cfg, model=model)
    elif cfg.dataset_type == 'activation':
        activation_dataset = PresaveActivationDataset(cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")

    lorsa = LowRankSparseAttention.from_pretrained(cfg.lorsa_dir)

    pbar = tqdm(total=cfg.total_tokens, desc="Evaluating Progress", unit="tokens", dynamic_ncols=True)
    sampled_tokens = 0
    step = 0

    ev_list = []
    l1_list = []
    mse_loss_list = []
    ground_truth_norm_list = []
    reconstructed_norm_list = []
    error_norm_list = []

    if lorsa.cfg.mode == "top_k":
        head_use_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
        head_positive_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
        head_negative_count = torch.zeros(cfg.lorsa_config.n_ov_heads, device=cfg.lorsa_config.device)
            
    while sampled_tokens < cfg.total_tokens:
        # get act
        hook_in, hook_out, filter_mask = activation_dataset.next(batch_size=cfg.batch_size)
        # hook_in, hook_out = lorsa.scale_norm(hook_in, hook_out)
        if lorsa.cfg.mode == "top_k":
            # top_k_z: (batch_size, query_pos, n_heads)
            out, top_k_z, l1 = lorsa.forward_top_k(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss
        elif lorsa.cfg.mode == "l1":
            out, l1 = lorsa.forward_l1(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss + cfg.l1_coef * l1[filter_mask].sum(dim=-1).mean()
        elif lorsa.cfg.mode == "default":
            out = lorsa.forward(hook_in)
            mse_loss = F.mse_loss(out[filter_mask], hook_out[filter_mask])
            loss = mse_loss

        # update head info
        sampled_tokens += filter_mask.sum().item()
        if lorsa.cfg.mode == "top_k":
            if cfg.lorsa_config.d_ov_head == 1:
                top_k_mask = (top_k_z.squeeze(dim=-1) != 0.).to(torch.int32)
            else:
                raise NotImplementedError
            counts = torch.sum(top_k_mask[filter_mask], dim=0)
            head_use_count += counts
            head_positive_count += torch.sum((top_k_z[filter_mask] > 0).to(torch.int32), dim=0)
            head_negative_count += torch.sum((top_k_z[filter_mask] < 0).to(torch.int32), dim=0)
        step += 1
        
        # calculate explained variance
        per_token_l2_loss = (
            (out[filter_mask] - hook_out[filter_mask]).pow(2).sum(dim=-1)
        )
        total_variance = (
            (out[filter_mask] - out[filter_mask].mean(0)).pow(2).sum(dim=-1)
        )
        explained_variance = 1 - per_token_l2_loss / total_variance

        ev_list.append(explained_variance.mean().cpu())
        mse_loss_list.append(mse_loss.cpu())
        l1_list.append(l1.sum(dim=-1).mean().cpu())
        ground_truth_norm_list.append(torch.norm(hook_out[filter_mask], p=2, dim=-1).mean().cpu())
        reconstructed_norm_list.append(torch.norm(out[filter_mask], p=2, dim=-1).mean().cpu())
        error_norm_list.append(torch.mean(torch.norm(out[filter_mask] - hook_out[filter_mask], p=2, dim=-1)).cpu())

        # update tqdm bar
        pbar.update(filter_mask.sum().item()) 
        pbar.set_postfix({
            "mse_loss": round(torch.tensor(mse_loss_list).mean().item(), 5), 
            "explained_variance": round(torch.tensor(ev_list).mean().item(), 2), 
            **({"l1": torch.tensor(l1_list).mean().item()} if lorsa.cfg.mode == "top_k" or lorsa.cfg.mode == "l1" else {})})
        pbar.refresh()

    pbar.close()

    positivity = torch.mean(torch.max(head_positive_count, head_negative_count) / torch.clamp((head_positive_count + head_negative_count), min=1))

    metrics = {
        "mse_loss": round(torch.tensor(mse_loss_list).mean().item(), 5), 
        "explained_variance": round(torch.tensor(ev_list).mean().item(), 5), 
        **({"l1": round(torch.tensor(l1_list).mean().item(), 2)} if lorsa.cfg.mode == "top_k" or lorsa.cfg.mode == "l1" else {}),
        'ground_truth_norm': round(torch.tensor(ground_truth_norm_list).mean().item(), 3),
        'reconstructed_norm': round(torch.tensor(reconstructed_norm_list).mean().item(), 3),
        'error_norm': round(torch.tensor(error_norm_list).mean().item(), 4),
        **({"sparsity/below 1e-5": (head_use_count / sampled_tokens < 1e-5).sum().item()} if lorsa.cfg.mode == "top_k" else {}),
        **({"sparsity/below 1e-6": (head_use_count / sampled_tokens < 1e-6).sum().item()} if lorsa.cfg.mode == "top_k" else {}),
        **({"positivity": round(positivity[positivity != 0].mean().item(), 5)} if lorsa.cfg.mode == "top_k" else {}),
        'head_use_frequency': {i: (head_use_count[i * int(lorsa.cfg.n_ov_heads / lorsa.cfg.n_qk_heads):(i + 1) * int(lorsa.cfg.n_ov_heads / lorsa.cfg.n_qk_heads)] != 0).sum().item() for i in range(lorsa.cfg.n_qk_heads)},
    }

    output_path = os.path.join(lorsa_dir, 'metrics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Metrics saved to {output_path}")
