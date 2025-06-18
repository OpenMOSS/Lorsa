from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field, fields

import os
import json
import torch

from transformers import AutoConfig, AutoTokenizer
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.loading_from_pretrained import convert_hf_model_config

from utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)

@dataclass(kw_only=True)
class LorsaConfig:
    # self attention head config
    d_qk_head: int
    d_ov_head: int
    n_qk_heads: int
    n_ov_heads: int
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: torch.dtype = torch.float32
    virtual_kv_num: int = 0
    use_z_relu: bool = False
    n_ctx: int = 256
    layer: int | None = None
    model_name: str | None = None
    use_flash_lorsa: bool = False
    
    mode: Literal["top_k", "jumprelu"] = "top_k"
    top_k: Optional[int] = None
    init_jumprelu_threshold: Optional[float] = 0.1
    
    avg_norm: dict = None
    
    # config from model config
    d_model: Optional[int] = None
    attn_scale: Optional[float] = None
    positional_embedding_type: Literal["default", "rotary"] = "rotary"
    rotary_scale: int = 1
    rotary_dim: Optional[int] = None
    rotary_base: Optional[int] = 10000
    rotary_adjacent_pairs: Optional[bool] = False
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_low_freq_factor: Optional[float] = None
    NTK_by_parts_high_freq_factor: Optional[float] = None
    NTK_by_parts_factor: Optional[float] = None
    old_context_len: Optional[int] = None

    def update_from_model_config(self, model_cfg: HookedTransformerConfig):
        self.d_model = model_cfg.d_model
        self.attn_scale = model_cfg.attn_scale
        self.positional_embedding_type = model_cfg.positional_embedding_type
        self.rotary_base = model_cfg.rotary_base
        self.rotary_adjacent_pairs = model_cfg.rotary_adjacent_pairs
        self.use_NTK_by_parts_rope = model_cfg.use_NTK_by_parts_rope
        if self.use_NTK_by_parts_rope:
            self.NTK_by_parts_low_freq_factor = model_cfg.NTK_by_parts_low_freq_factor
            self.NTK_by_parts_high_freq_factor = model_cfg.NTK_by_parts_high_freq_factor
            self.NTK_by_parts_factor = model_cfg.NTK_by_parts_factor
            self.old_context_len = model_cfg.old_context_len
    
    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs):
        d = {k: v for k, v in d.items() if k in [field.name for field in fields(cls)]}
        return cls(**d, **kwargs)
    
    def save_config(self, save_path: str):
        assert os.path.isdir(save_path), f"{save_path} must be a directory."
        d = self.to_dict()

        for k, v in d.items():
            if isinstance(v, torch.dtype):
                d[k] = convert_torch_dtype_to_str(v)

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(d, f, indent=4)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load the LorsaConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/lm_config.json.

        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the LorsaConfig constructor.
        """
        with open(os.path.join(path, "config.json"), "r") as f:
            lorsa_config = json.load(f)
        
        lorsa_config['dtype'] = convert_str_to_torch_dtype(lorsa_config['dtype'])

        return cls.from_dict(lorsa_config, **kwargs)
    
    def __post_init__(self):
        if self.n_ov_heads % self.n_qk_heads != 0:
            raise ValueError("n_ov_heads must be divisible by n_qk_heads")
        if self.top_k and self.top_k > self.n_ov_heads:
            raise ValueError("top_k must be less than or equal to n_ov_heads")

@dataclass(kw_only=True)
class LorsaTrainConfig:
    # lorsa config
    lorsa_config: LorsaConfig

    # init config
    init_qk_with_orig_qk: bool = False
    fix_qk: bool = False
    
    # dataset config
    dataset_path: str
    dataset_type: Literal["text", "activation"]
    num_workers: int
    prefetch_factor: int
    lm_batch_size: int
    buffer_size: int
    
    # training config
    batch_size: int
    total_tokens: int
    learning_rate: float
    final_learning_rate: float
    lr_warm_up_tokens: int
    lr_cool_down_tokens: int
    clip_grad_norm: float
    mode: Literal["top_k", "jumprelu"] = "top_k"
    init_scale_parameters: bool = True
    
    # k config
    k_scheduler_name: Literal['linear', 'exponential', 'cosine', 'smooth_step', 'sqrt']
    start_k: Optional[int] = None
    end_k: Optional[int] = None
    k_warm_up_tokens: Optional[int] = None
        
    # jumprelu config
    lambda_s: Optional[float] = None
    lambda_p: Optional[float] = None
    sparsity_c: Optional[float] = None
    lambda_s_final: Optional[float] = None  # lambda_s的最终值，用于线性warmup
    jumprelu_epsilon: Optional[float] = 2.0  # JumpReLU的epsilon参数
    
    # orig attention head config
    model_name: str
    model: str = None
    layer: int
    prepend_bos: bool = True
    
    # wandb config
    log_to_wandb: bool
    project_name: Optional[str] = None
    log_frequency: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # result config
    result_dir: str

    def update_lorsa_cfg_with_model_cfg(self):
        if self.model_name.startswith("NeelNanda"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                local_files_only=True,
            )
            model = HookedTransformer.from_pretrained_no_processing(
                self.model_name,
                use_flash_attn=True, 
                tokenizer=tokenizer,
                device=self.lorsa_config.device,
                dtype=self.lorsa_config.dtype,
            )
            tl_config = model.cfg
        else:
            hf_config = AutoConfig.from_pretrained(
                self.model if self.model is not None else self.model_name
            )
            tl_config = convert_hf_model_config(
                model_name=self.model_name,
                hf_config=hf_config,
            )
        self.lorsa_config.update_from_model_config(
            HookedTransformerConfig.unwrap(tl_config)
        )
    
    def __post_init__(self):
        self.result_dir = os.path.join(
            self.result_dir, 
            f"{self.wandb_project}/{self.project_name}"
        )
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


        # pass necessary configurations to lorsa config
        self.lorsa_config.mode = self.mode
        self.lorsa_config.layer = self.layer
        self.lorsa_config.model_name = self.model_name

        self.update_lorsa_cfg_with_model_cfg()


@dataclass(kw_only=True)
class LorsaEvaluateConfig:
    # lorsa config
    lorsa_dir: str
    lorsa_config: LorsaConfig
    
    # dataset config
    dataset_path: str
    dataset_type: Literal["text", "activation"]
    num_workers: int
    prefetch_factor: int
    lm_batch_size: int
    buffer_size: int
    
    # evaluating config
    batch_size: int
    total_tokens: int
    
    # orig attention head config
    model_name: str = None
    layer: int
    prepend_bos: bool = None

@dataclass(kw_only=True)
class LorsaAnalyzeConfig:
    # lorsa config
    lorsa_config: LorsaConfig
    
    # dataset config
    dataset_path: str
    dataset_type: Literal["text", "activation"]
    num_workers: int
    lm_batch_size: int
    buffer_size: int
    
    # orig attention head config
    model_name: str
    layer: int
    prepend_bos: bool
    
    # db config
    mongo_uri: str
    mongo_db: str

@dataclass(kw_only=True)
class DataGenConfig:
    # base config
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16

    # parallel config
    num_proc: int

    # dataset config
    dataset_path: str
    
    # model config
    model_name: str
    model: str
    use_flash_attn: bool

    # data config
    names_filter: list = None
    num_shards: int
    shard_index: int
    batch_size: int
    n_ctx: int
    prepend_bos: bool

    # result config
    result_dir: str
    
    # other config
    remove_bos_component: bool = False
    layers_to_remove_bos: list = None