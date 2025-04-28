import os
import sys
sys.path.append('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/src')
from typing import List, Set, Tuple, Dict, Union
from jaxtyping import Float, Int, Bool
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    ChameleonForConditionalGeneration,
    PreTrainedModel,
)
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components import Attention
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
torch.set_grad_enabled(False)

import copy

from tqdm import tqdm

import numpy as np
import einops

from models.lorsa import LowRankSparseAttention
from config import LorsaTrainConfig, LorsaConfig
from analysis.new_analysis import sample_max_activating_sequences

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    ActivationFactory,
    AnalyzeSAESettings,
    SAEConfig,
    SparseAutoEncoder,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    analyze_sae,
)

parser = argparse.ArgumentParser(description='Process hyparameters')
parser.add_argument('--layer', type=int, required=False, default=0, help='Layer number')
parser.add_argument('--head_index', type=int, required=False, default=0, help='Head index')
args = parser.parse_args()

model_name = "EleutherAI/pythia-160m"
seq_len=256
device='cuda'

def load_model(model_name: str):
    device = 'cuda'
    model_path = {
        "meta-llama/Llama-3.1-8B": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/Llama-3.1-8B",
        "EleutherAI/pythia-160m": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/pythia-160m",
    }[model_name]
    
    dtype = {
        "meta-llama/Llama-3.1-8B": torch.bfloat16,
        "EleutherAI/pythia-160m": torch.float16,
    }[model_name]

    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
        local_files_only=True,
    )
    hf_processor = None

    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        use_flash_attn=False,
        device=device,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        processor=hf_processor,
        dtype=dtype,
        hf_config=hf_model.config,
    )
    model.eval()
    return model, hf_tokenizer

model, tokenizer = load_model(model_name)
model.eval()
for param in model.parameters():
    param.requires_grad = False

dtype = {
    "meta-llama/Llama-3.1-8B": torch.bfloat16,
    "EleutherAI/pythia-160m": torch.float16,
}[model_name]

ignore_tokens = {
    model.tokenizer.bos_token_id,
    model.tokenizer.eos_token_id,
    model.tokenizer.pad_token_id,
}

@torch.no_grad()
def get_activation_with_filter_mask(
    model: HookedTransformer,
    batch: List[str],
    ignore_tokens: Set[int],
    cfg: LorsaConfig,
    seq_len: int = 64,
) -> Tuple[
    Float[torch.Tensor, "batch_size ctx_length d_model"],
    Bool[torch.Tensor, "batch_size ctx_length"],
]:
    tokens = model.to_tokens(
        batch, 
        prepend_bos=True,
    ).to(cfg.device, non_blocking=True)

    tokens = tokens[:, :seq_len]

    if len(ignore_tokens) > 0:
        filter_mask = torch.any(
            torch.stack(
                [tokens == ignore_token for ignore_token in ignore_tokens], dim=0
            ),
            dim=0,
        )  # This gives True on ignore tokens and False on informative ones.

    hook_in_name = f'blocks.{cfg.layer}.ln1.hook_normalized'

    _, cache = model.run_with_cache(tokens, names_filter=[hook_in_name])
    hook_in = cache[hook_in_name]

    return hook_in, ~filter_mask

layer = args.layer
lorsa_path = f'/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/result/pythia-160m/oneway_all_layer_result/L{layer}A'
lorsa = LowRankSparseAttention.from_pretrained(
    lorsa_path,
    device='cuda',
    dtype=dtype
)
lorsa.fold_W_O_into_W_V()

sae_dir = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Language-Model-SAEs/result/Pythia-160m-SAE-for-Lorsa"
sae_path = os.path.join(sae_dir, f"L{layer}AIN")
sae_cfg=SAEConfig.from_pretrained(
    pretrained_name_or_path=sae_path,
    device="cuda",
    dtype=dtype,
)
sae = SparseAutoEncoder.from_config(sae_cfg)

# load dataset
dataset_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/data/SlimPajama-3B'
dataset = load_from_disk(dataset_path)

batch_size = 16
dataloader = torch.utils.data.DataLoader(
    dataset['text'], 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)
data_iter = iter(dataloader)

qk_head_index = args.head_index
max_feature_mse = torch.zeros([sae.cfg.d_sae + 2], dtype=dtype).to(device)
total_sampled_sentences = 16 * 256
sampled_sentences = 0

with tqdm(total=total_sampled_sentences, initial=0, unit='sentences') as pbar:
    while sampled_sentences < total_sampled_sentences:
        batch = next(data_iter)
        hook_in, filter_mask = get_activation_with_filter_mask(model=model, batch=batch, ignore_tokens=ignore_tokens, cfg=lorsa.cfg, seq_len=seq_len)
        hook_in = hook_in.to(dtype)
        head_act_mask = filter_mask
        head_act_index = torch.nonzero(head_act_mask)

        for i in range(head_act_index.shape[0]):
            batch_index = head_act_index[i][0]
            q_index = head_act_index[i][1]
            q_resid = hook_in[batch_index, q_index]
            k_resid = hook_in[batch_index, :q_index+1]
            sae_norm_factor = sae.compute_norm_factor(hook_in, hook_point=f'blocks.{layer}.ln1.hook_normalized')
            k_resid_feature_acts = sae.encode(k_resid) / sae_norm_factor
            k_resid_feature_tensors = (k_resid_feature_acts.T.unsqueeze(dim=2) * sae.decoder.weight.T.unsqueeze(dim=1))
            k_resid_feature_tensors = torch.cat([k_resid_feature_tensors, sae.decoder.bias.repeat([q_index+1, 1]).unsqueeze(0)], dim=0)
            k_resid_feature_tensors = torch.cat([k_resid_feature_tensors, torch.zeros_like(k_resid_feature_tensors[:1])], dim=0)
            k_resid_feature_tensors = k_resid - k_resid_feature_tensors
            q = q_resid @ model.blocks[layer].attn.W_Q[qk_head_index] + model.blocks[layer].attn.b_Q[qk_head_index]
            k = k_resid_feature_tensors @ model.blocks[layer].attn.W_K[qk_head_index] + model.blocks[layer].attn.b_K[qk_head_index]
            q = model.blocks[layer].attn.apply_rotary(q.reshape(1, 1, 1, q.shape[0]).repeat([1, q_index+1, 1, 1])).squeeze()[q_index]
            k = model.blocks[layer].attn.apply_rotary(k.reshape(k.shape[0], k.shape[1], 1, k.shape[2])).squeeze()
            feature_attention_scores = k @ q / model.blocks[layer].attn.cfg.attn_scale
            feature_pattern = torch.softmax(feature_attention_scores, dim=1)
            attention_scores = feature_attention_scores[-1]
            pattern = feature_pattern[-1]
            feature_mse = (feature_pattern - pattern).norm(p=2, dim=1)
            max_feature_mse = torch.maximum(max_feature_mse, feature_mse)
        pbar.update(batch_size)
        sampled_sentences += batch_size
        
torch.save(max_feature_mse, f'/inspire/hdd/global_user/hezhengfu-240208120186/jx_files/monosemanticity/k/result/pythia-160m/orig_head/L{layer}-{qk_head_index}.pt')