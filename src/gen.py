import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformer_lens import HookedTransformer
from datasets import load_from_disk
from tqdm import tqdm

from config import DataGenConfig

@torch.no_grad()
def generate_train_data(cfg: DataGenConfig):
    # load model
    hf_model = GPTNeoXForCausalLM.from_pretrained(
        cfg.model, 
        local_files_only=True
    )
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
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
    filtered_dataset = dataset.filter(
        lambda example: len(tokenizer.encode(example['text'])) >= cfg.n_ctx,
        num_proc = cfg.num_proc,
    )
    # create DataLoader
    data = DataLoader(filtered_dataset['text'], batch_size=cfg.batch_size, num_workers=cfg.num_proc, shuffle=False)
    data_iter = iter(data)

    with ProcessPoolExecutor(max_workers=None) as executor:
        for batch_index in tqdm(range(min(cfg.n_batchs, len(filtered_dataset) // cfg.batch_size))):
            try:
                batch = next(data_iter)
            except StopIteration:
                print('stop iteration')
                exit(0)
            tokens = model.to_tokens(batch, prepend_bos=True).to(cfg.device)[:, :cfg.n_ctx]
            filter_mask = torch.logical_and(tokens.ne(model.tokenizer.eos_token_id), tokens.ne(model.tokenizer.pad_token_id))
            filter_mask = torch.logical_and(filter_mask, tokens.ne(model.tokenizer.bos_token_id))

            _, cache = model.run_with_cache_until(tokens, names_filter=([f'blocks.{l}.ln1.hook_normalized' for l in cfg.layers] + [f'blocks.{l}.hook_attn_out' for l in cfg.layers]), until=f'blocks.{max(cfg.layers)}.hook_attn_out')

            tasks = []
            result_dir = os.path.join(cfg.result_dir, 'filter_mask')
            tasks.append((filter_mask, result_dir, f'{batch_index}.pt'))

            for key, value in cache.items():
                result_dir = os.path.join(cfg.result_dir, key)
                tasks.append((value.to(cfg.dtype), result_dir, f'{batch_index}.pt'))

            futures = [executor.submit(save_tensor, task) for task in tasks]

            for future in futures:
                future.result()

def save_tensor(tensor_info):
    tensor, dir_path, filename = tensor_info
    os.makedirs(dir_path, exist_ok=True)
    torch.save(tensor, os.path.join(dir_path, filename))