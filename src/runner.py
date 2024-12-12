import os

import torch
from torch.utils.data import DataLoader

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from transformer_lens import HookedTransformer

from datasets import load_from_disk

from model.attention import LowRankSparseAttention
from config import LoRSATrainConfig, DataGenConfig
from train import train_lorsa, train_lorsa_without_forward

import wandb

import copy

def train_lorsa_without_forward_runner(cfg: LoRSATrainConfig):
    # load model
    hf_model = GPTNeoXForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(cfg.model_name)
    model = HookedTransformer.from_pretrained(cfg.model_name, hf_model=hf_model, tokenizer=tokenizer, device=cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    orig_attn = model.blocks[cfg.layer].attn
    
    cfg.lorsa_config.update_from_model_config(model.cfg)
    
    lorsa = LowRankSparseAttention(cfg.lorsa_config).to(cfg.lorsa_config.device)
    
    lorsa.initialize_parameters(b_O = orig_attn.b_O.data.clone().detach().to(cfg.lorsa_config.dtype))

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
            save_code=False,
            sync_tensorboard=False,
            settings=wandb.Settings(
                _disable_stats=True
            )
        )
        
    # train sah
    lorsa = train_lorsa_without_forward(
        lorsa=lorsa,
        cfg=cfg,
    )
    
    
    # save
    result_dir = os.path.join(cfg.result_dir, f"{cfg.wandb_project}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    torch.save(lorsa.state_dict(), os.path.join(result_dir, f"{cfg.project_name}.pth"))
    
    # finish wandb
    if cfg.log_to_wandb:
        wandb.finish()


def train_lorsa_runner(cfg: LoRSATrainConfig):
    # load model
    hf_model = GPTNeoXForCausalLM.from_pretrained(cfg.lorsa_config.model_name)
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(cfg.lorsa_config.model_name)
    model = HookedTransformer.from_pretrained(cfg.lorsa_config.model_name, hf_model=hf_model, tokenizer=tokenizer, device=cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    
    # get original attention block
    orig_attn = model.blocks[cfg.lorsa_config.layer].attn
    
    # update sah config
    cfg.lorsa_config.update_from_model_config(model.cfg)
    
    # initialize sah
    sah = LowRankSparseAttention(cfg.lorsa_config).to(cfg.lorsa_config.device)
    sah.initialize_parameters(b_O = orig_attn.b_O.data.clone().detach().to(cfg.lorsa_config.dtype))

    
    # load dataset
    dataset = load_from_disk(cfg.dataset_path)
    data = DataLoader(dataset['text'], batch_size=cfg.batch_size)
    
    # init wandb
    if cfg.log_to_wandb:
        config = copy.deepcopy(cfg)
        config.lorsa_config.dtype = str(config.lorsa_config.dtype)
        wandb.init(
            project=cfg.wandb_project,
            config=config,
            name=cfg.project_name,
            entity=cfg.wandb_entity,
        )
        
    # train sah
    sah = train_lorsa(
        sah=sah,
        model=model,
        cfg=cfg,
        data=data,
    )
    
    
    # save
    folder_path = f"/remote-home1/jxwang/project/monofuctional_attn/result/{cfg.wandb_project}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(sah.state_dict(), f"/remote-home1/jxwang/project/monofuctional_attn/result/{cfg.wandb_project}/{cfg.project_name}.pth")
    
    # finish wandb
    if cfg.log_to_wandb:
        wandb.finish()