import os
import math

import torch
from torch.utils.data import DataLoader

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer, AutoModelForCausalLM

from transformer_lens import HookedTransformer

from datasets import load_from_disk

from models.lorsa import LowRankSparseAttention
from config import LorsaTrainConfig, LorsaAnalyzeConfig, DataGenConfig
from train import train_lorsa, debug_train_lorsa
from activations import TextActivationDataset, PresaveActivationDataset
import wandb

import copy

def train_lorsa_runner(cfg: LorsaTrainConfig):
    if cfg.dataset_type == 'text' or cfg.init_qk_with_orig_qk == True:
        # load model
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            local_files_only=True
        )
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.model, 
            local_files_only=True
        )

        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, 
            use_flash_attn=True, 
            hf_model=hf_model,
            hf_config=hf_model.config,
            tokenizer=tokenizer,
            device=cfg.lorsa_config.device,
            dtype=cfg.lorsa_config.dtype,
        )
        model.offload_params_after(f'blocks.{cfg.layer}.hook_attn_out', torch.tensor([[0]], device=cfg.lorsa_config.device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
    # load activation dataset
    if cfg.dataset_type == 'text':
        activation_dataset = TextActivationDataset(cfg=cfg, model=model, tokenizer=tokenizer)
    elif cfg.dataset_type == 'activation':
        activation_dataset = PresaveActivationDataset(cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")
    
    cfg.lorsa_config.avg_norm = activation_dataset.cal_norm()

    cfg.lorsa_config.save_config(save_path=cfg.result_dir)

    lorsa = LowRankSparseAttention(cfg.lorsa_config).to(cfg.lorsa_config.device, non_blocking=True)

    if cfg.init_qk_with_orig_qk == True:
        orig_attn = model.blocks[cfg.layer].attn
        lorsa.initialize_parameters(
            W_Q = orig_attn.W_Q.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1, 1) * lorsa.cfg.avg_norm['in'] / math.sqrt(lorsa.cfg.d_model),
            b_Q = orig_attn.b_Q.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1),
            W_K = orig_attn.W_K.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1, 1) * lorsa.cfg.avg_norm['in'] / math.sqrt(lorsa.cfg.d_model),
            b_K = orig_attn.b_K.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1),
        )
        print('init lorsa qk parameters using orig qk')
        if cfg.fix_qk == True:
            lorsa.set_requires_grad('W_Q', requires_grad=False)
            lorsa.set_requires_grad('b_Q', requires_grad=False)
            lorsa.set_requires_grad('W_K', requires_grad=False)
            lorsa.set_requires_grad('b_K', requires_grad=False)

    # init wandb
    if cfg.log_to_wandb:
        config = copy.deepcopy(cfg)
        config.lorsa_config.dtype = str(config.lorsa_config.dtype)
        wandb.init(
            project=cfg.wandb_project,
            config=config,
            name=cfg.project_name,
            entity=cfg.wandb_entity,
            mode=os.getenv('WANDB_MODE', 'online'),
            # save_code=False,
            # ync_tensorboard=False,
            # settings=wandb.Settings(
            #     _disable_stats=True
            # )
        )
        
    # train lorsa
    lorsa = train_lorsa(
        lorsa=lorsa,
        cfg=cfg,
        activation_dataset=activation_dataset,
    )
    
    # save
    torch.save(lorsa.state_dict(), os.path.join(cfg.result_dir, f"final.pth"))
    
    # finish wandb
    if cfg.log_to_wandb:
        wandb.finish()

def debug_train_lorsa_runner(cfg: LorsaTrainConfig):
    if cfg.dataset_type == 'text' or cfg.init_qk_with_orig_qk == True:
        # load model
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            local_files_only=True
        )
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.model, 
            local_files_only=True
        )

        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, 
            use_flash_attn=True, 
            hf_model=hf_model,
            hf_config=hf_model.config,
            tokenizer=tokenizer,
            device=cfg.lorsa_config.device,
            dtype=cfg.lorsa_config.dtype,
        )
        model.offload_params_after(f'blocks.{cfg.layer}.hook_attn_out', torch.tensor([[0]], device=cfg.lorsa_config.device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
    # load activation dataset
    if cfg.dataset_type == 'text':
        activation_dataset = TextActivationDataset(cfg=cfg, model=model, tokenizer=tokenizer)
    elif cfg.dataset_type == 'activation':
        activation_dataset = PresaveActivationDataset(cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")
    
    cfg.lorsa_config.avg_norm = activation_dataset.cal_norm()

    lorsa = LowRankSparseAttention(cfg.lorsa_config).to(cfg.lorsa_config.device, non_blocking=True)

    if cfg.init_qk_with_orig_qk == True:
        orig_attn = model.blocks[cfg.layer].attn
        lorsa.initialize_parameters(
            W_Q = orig_attn.W_Q.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1, 1) * lorsa.cfg.avg_norm['in'] / math.sqrt(lorsa.cfg.d_model),
            b_Q = orig_attn.b_Q.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1),
            W_K = orig_attn.W_K.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1, 1) * lorsa.cfg.avg_norm['in'] / math.sqrt(lorsa.cfg.d_model),
            b_K = orig_attn.b_K.repeat(lorsa.cfg.n_qk_heads // orig_attn.cfg.n_heads, 1),
        )
        print('init lorsa qk parameters using orig qk')
        if cfg.fix_qk == True:
            lorsa.set_requires_grad('W_Q', requires_grad=False)
            lorsa.set_requires_grad('b_Q', requires_grad=False)
            lorsa.set_requires_grad('W_K', requires_grad=False)
            lorsa.set_requires_grad('b_K', requires_grad=False)

    # init wandb
    if cfg.log_to_wandb:
        config = copy.deepcopy(cfg)
        config.lorsa_config.dtype = str(config.lorsa_config.dtype)
        wandb.init(
            project=cfg.wandb_project,
            config=config,
            name=cfg.project_name,
            entity=cfg.wandb_entity,
            mode=os.getenv('WANDB_MODE', 'online'),
            # save_code=False,
            # ync_tensorboard=False,
            # settings=wandb.Settings(
            #     _disable_stats=True
            # )
        )
        
    # train lorsa
    lorsa = debug_train_lorsa(
        lorsa=lorsa,
        cfg=cfg,
        activation_dataset=activation_dataset,
    )