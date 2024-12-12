from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass

import os

import torch

from transformer_lens import HookedTransformerConfig

@dataclass(kw_only=True)
class LoRSAConfig:
    # self attention head config
    d_qk_head: int
    d_ov_head: int
    n_qk_heads: int
    n_ov_heads: int
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: torch.dtype = torch.float32
    virtual_kv_num: int = 0
    use_z_relu: bool = False
    
    mode: Literal["default", "top_k", "l2"] = "default"
    top_k: Optional[int] = None
    
    # config from model config
    d_model: Optional[int] = None
    attn_scale: Optional[float] = None
    n_ctx: Optional[int] = None
    positional_embedding_type: Literal["default", "rotary"] = "rotary"
    rotary_scale: int = 1
    rotary_dim: Optional[int] = None
    rotary_base: Optional[int] = 10000
    rotary_adjacent_pairs: Optional[bool] = False

    def update_from_model_config(self, model_cfg: HookedTransformerConfig):
        self.d_model = model_cfg.d_model
        self.attn_scale = model_cfg.attn_scale
        self.n_ctx = model_cfg.n_ctx
        self.positional_embedding_type = model_cfg.positional_embedding_type
        self.rotary_base = model_cfg.rotary_base
        self.rotary_adjacent_pairs = model_cfg.rotary_adjacent_pairs
        
    
    def __post_init__(self):
        if self.n_ov_heads % self.n_qk_heads != 0:
            raise ValueError("n_ov_heads must be divisible by n_qk_heads")
        if self.top_k > self.n_ov_heads:
            raise ValueError("top_k must be less than or equal to n_ov_heads")

@dataclass(kw_only=True)
class LoRSATrainConfig:
    # lorsa config
    lorsa_config: LoRSAConfig
    
    # dataset config
    dataset_path: str
    num_workers: int
    
    # training config
    batch_size: int
    total_tokens: int
    learning_rate: float
    final_learning_rate: float
    lr_warm_up_tokens: int
    lr_cool_down_tokens: int
    device: Literal["cpu", "cuda"] = 'cuda'
    mode: Literal["default", "top_k", "l2"] = "default"
    init_scale_parameters: bool = True
    
    # k config
    k_scheduler_name: Literal['linear', 'exponential', 'cosine', 'smooth_step', 'sqrt']
    start_k: Optional[int] = None
    end_k: Optional[int] = None
    k_warm_up_tokens: Optional[int] = None
    
    # l2 config
    l2_coef: Optional[float] = None
    
    # orig attention head config
    model_name: str
    layer: int
    max_length: int
    prepend_bos: bool
    
    # wandb config
    log_to_wandb: bool
    project_name: Optional[str] = None
    log_frequency: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # result config
    result_dir: str
    
    def __post_init__(self):
        self.lorsa_config.mode = self.mode

@dataclass(kw_only=True)
class DataGenConfig:
    # base config
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16

    # dataset config
    dataset_path: str
    
    # model config
    model_name: str
    use_flash_attn: bool

    # data config
    layers: []
    n_batchs: int
    batch_size: int
    max_length: int
    prepend_bos: bool

    # result config
    result_dir: str