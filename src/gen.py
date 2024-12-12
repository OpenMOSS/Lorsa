import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformer_lens import HookedTransformer
from datasets import load_from_disk
from tqdm import tqdm

from config import DataGenConfig

def generate_train_data(cfg: DataGenConfig):
    # load model
    hf_model = GPTNeoXForCausalLM.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    hf_tokenizer = GPTNeoXTokenizerFast.from_pretrained(cfg.model_name)
    model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name,
            use_flash_attn=cfg.use_flash_attn,
            device=cfg.device,
            hf_model=hf_model,
            tokenizer=hf_tokenizer,
            dtype=cfg.dtype,
    )
    # load dataset
    dataset = load_from_disk(cfg.dataset_path)
    data = DataLoader(dataset['text'], batch_size=1, num_workers=4, shuffle=False)
    data_iter = iter(data)

    # generate

    for batch_index in tqdm(range(cfg.n_batchs)):
        batch_tokens = torch.empty(cfg.batch_size, cfg.max_length, dtype=torch.int32, device=cfg.device)

        sentence_index = 0
        while sentence_index != cfg.batch_size:
            batch = next(data_iter)
            tokens = model.to_tokens(batch, prepend_bos=cfg.prepend_bos).to(cfg.device)
            if tokens.shape[1] < cfg.max_length:
                continue
            batch_tokens[sentence_index] = tokens[:, :cfg.max_length].unsqueeze(0)
            sentence_index += 1

        filter_mask = torch.logical_and(batch_tokens.ne(model.tokenizer.eos_token_id), batch_tokens.ne(model.tokenizer.pad_token_id))
        filter_mask = torch.logical_and(filter_mask, batch_tokens.ne(model.tokenizer.bos_token_id))

        _, cache = model.run_with_cache_until(batch_tokens, names_filter=([f'blocks.{l}.ln1.hook_normalized' for l in cfg.layers] + [f'blocks.{l}.hook_attn_out' for l in cfg.layers]), until=f'blocks.{max(cfg.layers)}.hook_attn_out')

        def save_tensor(tensor, dir_path, filename):
            os.makedirs(dir_path, exist_ok=True)
            torch.save(tensor, os.path.join(dir_path, filename))

        with ThreadPoolExecutor() as executor:
            result_dir = os.path.join(cfg.result_dir, f'filter_mask')
            executor.submit(save_tensor, filter_mask, result_dir, f'{batch_index}.pt')
            
            for key, value in cache.items():
                result_dir = os.path.join(cfg.result_dir, key)
                executor.submit(save_tensor, value.to(cfg.dtype), result_dir, f'{batch_index}.pt')
        
        '''
        # 保存 cache
        for key, value in cache.items():
            result_dir = os.path.join(cfg.result_dir, key)
            executor.submit(save_tensor, value.to(cfg.dtype), result_dir, f'{batch_index}.pt')

            # save
            result_dir = os.path.join(cfg.result_dir, f'filter_mask')
            os.makedirs(result_dir, exist_ok=True)
            torch.save(filter_mask, os.path.join(result_dir, f'{batch_index}.pt'))
            for key, value in cache.items():
                value = value.to(cfg.dtype)
                result_dir = os.path.join(cfg.result_dir, key)
                os.makedirs(result_dir, exist_ok=True)
                torch.save(value, os.path.join(result_dir, f"{batch_index}.pt"))
        '''