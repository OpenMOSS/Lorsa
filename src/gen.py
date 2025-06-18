import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_from_disk
from tqdm import tqdm

from config import DataGenConfig

@torch.no_grad()
def generate_train_data(cfg: DataGenConfig):
    # load model
    if cfg.model_name.startswith("NeelNanda"):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            local_files_only=True,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name,
            use_flash_attn=True, 
            tokenizer=tokenizer,
            device=cfg.device,
            dtype=cfg.dtype,
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model, 
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, 
            local_files_only=True
        )

        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, 
            use_flash_attn=True, 
            hf_model=hf_model,
            hf_config=hf_model.config,
            tokenizer=tokenizer,
            device=cfg.device,
            dtype=cfg.dtype,
        )
    # load dataset
    dataset = load_from_disk(cfg.dataset_path)
    dataset = dataset.shard(num_shards=cfg.num_shards, index=cfg.shard_index)
    filtered_dataset = dataset.filter(
        lambda example: len(tokenizer.encode(example['text'], max_length=cfg.n_ctx, truncation=True)) >= cfg.n_ctx,
        num_proc = cfg.num_proc,
    )
    # create DataLoader
    data = DataLoader(filtered_dataset['text'], batch_size=cfg.batch_size, num_workers=cfg.num_proc, shuffle=False)
    data_iter = iter(data)

    with ThreadPoolExecutor(max_workers=None) as executor:
        for batch_index in tqdm(range(len(filtered_dataset) // cfg.batch_size)):
            try:
                batch = next(data_iter)
            except StopIteration:
                print('stop iteration')
                exit(0)
            if cfg.remove_bos_component:
                tokens = model.to_tokens(batch, prepend_bos=True).to(cfg.device)[:, :cfg.n_ctx+1]
            else:
                tokens = model.to_tokens(batch, prepend_bos=True).to(cfg.device)[:, :cfg.n_ctx]
            filter_mask = torch.logical_and(tokens.ne(model.tokenizer.eos_token_id), tokens.ne(model.tokenizer.pad_token_id))
            filter_mask = torch.logical_and(filter_mask, tokens.ne(model.tokenizer.bos_token_id))
            
            if cfg.remove_bos_component:
                add_names = [f'blocks.{l}.{suffix}' for l in cfg.layers_to_remove_bos for suffix in ['attn.hook_pattern', 'attn.hook_v']]
                _, cache = model.run_with_cache_until(tokens, names_filter=cfg.names_filter+add_names, until=cfg.names_filter[-1])
                for l in cfg.layers_to_remove_bos:
                    attn_pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [batch, head_index, query_pos, key_pos]
                    attn_v = cache[f'blocks.{l}.attn.hook_v']  # [batch, key_pos, head_index, d_head]
                    
                    # [batch, head_index, query_pos]
                    bos_weights = attn_pattern[:, :, :, 0]
                    
                    # [batch, head_index, d_head]
                    bos_values = attn_v[:, 0, :, :]
                    
                    z = bos_weights.unsqueeze(-1) * bos_values.unsqueeze(2)  # [batch, head_index, query_pos, d_head]
                    
                    w_o = model.blocks[l].attn.W_O  # [head_index, d_head, d_model]
                    
                    # z: [batch, head_index, query_pos, d_head]
                    # w_o: [head_index, d_head, d_model]
                    bos_output = torch.einsum('bnqh,nhd->bqd', z, w_o)  # [batch, query_pos, d_model]
                    
                    cache[f'blocks.{l}.hook_attn_out'] -= bos_output
                    
                tasks = []
                result_dir = os.path.join(cfg.result_dir, 'filter_mask')
                tasks.append((filter_mask[:, 1:].cpu(), result_dir, f"shard-{cfg.shard_index:04d}-chunk-{batch_index:06d}.pt"))

                for key in cfg.names_filter:
                    result_dir = os.path.join(cfg.result_dir, key)
                    tasks.append((cache[key][:, 1:].to(cfg.dtype).cpu(), result_dir, f"shard-{cfg.shard_index:04d}-chunk-{batch_index:06d}.pt"))

                futures = [executor.submit(save_tensor, task) for task in tasks]

                for future in futures:
                    future.result()
                
            else:
                _, cache = model.run_with_cache_until(tokens, names_filter=cfg.names_filter, until=cfg.names_filter[-1])

                tasks = []
                result_dir = os.path.join(cfg.result_dir, 'filter_mask')
                tasks.append((filter_mask.cpu(), result_dir, f"shard-{cfg.shard_index:04d}-chunk-{batch_index:06d}.pt"))

                for key, value in cache.items():
                    result_dir = os.path.join(cfg.result_dir, key)
                    tasks.append((value.to(cfg.dtype).cpu(), result_dir, f"shard-{cfg.shard_index:04d}-chunk-{batch_index:06d}.pt"))

                futures = [executor.submit(save_tensor, task) for task in tasks]

                for future in futures:
                    future.result()

def save_tensor(tensor_info):
    tensor, dir_path, filename = tensor_info
    os.makedirs(dir_path, exist_ok=True)
    torch.save(tensor, os.path.join(dir_path, filename))