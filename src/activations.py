import os
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, load_from_disk

from transformer_lens import HookedTransformer

from tqdm import tqdm

from config import LorsaTrainConfig


class MultiKeyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, keys: list, dtypes: list):
        self.dataset = dataset
        self.keys = keys
        self.dtypes = dtypes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        data = [torch.tensor(raw_data[k], dtype=dt) for k, dt in zip(self.keys, self.dtypes)]
        return tuple(data)

class ActivationDataset():
    def __init__(self, cfg: LorsaTrainConfig, model: HookedTransformer):
        self.hook_in_name = f'blocks.{cfg.layer}.ln1.hook_normalized'
        # self.hook_in_name = f'blocks.{cfg.layer}.hook_resid_pre'
        self.hook_out_name = f'blocks.{cfg.layer}.hook_attn_out'
        self.model = model
        self.cfg = cfg
    
    
    @torch.no_grad()
    def cal_norm(self):
        hook_in, hook_out, filter_mask = self.next(batch_size = self.cfg.buffer_size)
        hook_in_norm = hook_in[filter_mask].norm(p=2, dim=1).mean().item()
        hook_out_norm = hook_out[filter_mask].norm(p=2, dim=1).mean().item()
        print(f"Average input activation norm: {hook_in_norm}\nAverage output activation norm: {hook_out_norm}")
        return {
            "in": hook_in_norm,
            "out": hook_out_norm,
        }
        
    def next(self, batch_size: int):
        if self.index < batch_size:
            self.fill()
        self.index -= batch_size

        input_tensor = self.act_buffer['input'][self.index:self.index+batch_size]
        output_tensor = self.act_buffer['output'][self.index:self.index+batch_size]
        filter_mask_tensor = self.act_buffer['filter_mask'][self.index:self.index+batch_size]

        return input_tensor, output_tensor, filter_mask_tensor

    
class TextActivationDataset(ActivationDataset):
    def __init__(self, cfg: LorsaTrainConfig, model: HookedTransformer, tokenizer):
        self.hook_in_name = f'blocks.{cfg.layer}.ln1.hook_normalized'
        # self.hook_in_name = f'blocks.{cfg.layer}.hook_resid_pre'
        self.hook_out_name = f'blocks.{cfg.layer}.hook_attn_out'
        self.model = model
        self.cfg = cfg
        dataset = load_from_disk(cfg.dataset_path)
        filtered_dataset = dataset.filter(
            lambda example: len(tokenizer.encode(example['text'], max_length=cfg.lorsa_config.n_ctx, truncation=True)) >= cfg.lorsa_config.n_ctx,
            num_proc = cfg.num_workers,
        )
        self.dataloader = DataLoader(filtered_dataset['text'], batch_size=cfg.lm_batch_size, num_workers=cfg.num_workers)
        self.data_iter = iter(self.dataloader)
        self.act_buffer = {
            'input': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, self.cfg.lorsa_config.d_model, dtype=self.cfg.lorsa_config.dtype, device=self.cfg.lorsa_config.device),
            'output': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, self.cfg.lorsa_config.d_model, dtype=self.cfg.lorsa_config.dtype, device=self.cfg.lorsa_config.device),
            'filter_mask': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, dtype=torch.bool, device=self.cfg.lorsa_config.device)
        }
        self.index = 0
        self.fill()
    
    @torch.no_grad()
    def fill(self):
        while self.cfg.buffer_size - self.index >= self.cfg.lm_batch_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)
            tokens = self.model.to_tokens(batch, prepend_bos=self.cfg.prepend_bos).to(self.cfg.lorsa_config.device)
            tokens = tokens[:, :self.cfg.lorsa_config.n_ctx]
            _, cache = self.model.run_with_cache_until(tokens, names_filter=[self.hook_in_name, self.hook_out_name], until=self.hook_out_name)
            filter_mask = torch.logical_and(tokens.ne(self.model.tokenizer.eos_token_id), tokens.ne(self.model.tokenizer.pad_token_id))
            filter_mask = torch.logical_and(filter_mask, tokens.ne(self.model.tokenizer.bos_token_id))
            self.act_buffer['input'][self.index:self.index+self.cfg.lm_batch_size] = cache[self.hook_in_name].to(self.cfg.lorsa_config.dtype)
            self.act_buffer['output'][self.index:self.index+self.cfg.lm_batch_size] = cache[self.hook_out_name].to(self.cfg.lorsa_config.dtype)
            self.act_buffer['filter_mask'][self.index:self.index+self.cfg.lm_batch_size] = filter_mask
            self.index += self.cfg.lm_batch_size


class PresaveLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: LorsaTrainConfig):
        """
        Initialize the dataset with file paths and configuration.
        """
        self.hook_in_name = f'blocks.{cfg.layer}.ln1.hook_normalized'
        # self.hook_in_name = f'blocks.{cfg.layer}.hook_resid_pre'
        self.hook_out_name = f'blocks.{cfg.layer}.hook_attn_out'
        
        self.cfg = cfg
        
        self.file_paths = self.get_file_paths()

    def __len__(self):
        """
        Returns the number of batches available in the dataset.
        """
        return len(self.file_paths['input'])

    def __getitem__(self, index):
        """
        Load and return a batch of data.
        """
        input_tensor = torch.load(self.file_paths['input'][index], weights_only=True).to(self.cfg.lorsa_config.dtype).to(self.cfg.lorsa_config.device, non_blocking=True)
        output_tensor = torch.load(self.file_paths['output'][index], weights_only=True).to(self.cfg.lorsa_config.dtype).to(self.cfg.lorsa_config.device, non_blocking=True)
        filter_mask = torch.load(self.file_paths['filter_mask'][index], weights_only=True).to(self.cfg.lorsa_config.device, non_blocking=True)

        return {
            'input': input_tensor,
            'output': output_tensor,
            'filter_mask': filter_mask,
        }
    
    def get_file_paths(self):
        file_paths = {
            'input': [],
            'output': [],
            'filter_mask': []
        }
    
        input_file_paths = []
        input_file_dir = os.path.join(self.cfg.dataset_path, self.hook_in_name)
        for item in os.listdir(input_file_dir):
            item_path = os.path.join(input_file_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pt'):
                input_file_paths.append(item_path)
        input_file_paths.sort()
        file_paths['input'] += input_file_paths

        output_file_paths = []
        output_file_dir = os.path.join(self.cfg.dataset_path, self.hook_out_name)
        for item in os.listdir(output_file_dir):
            item_path = os.path.join(output_file_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pt'):
                output_file_paths.append(item_path)
        output_file_paths.sort()
        file_paths['output'] += output_file_paths

        filter_mask_file_paths = []
        filter_mask_file_dir = os.path.join(self.cfg.dataset_path, 'filter_mask')
        for item in os.listdir(filter_mask_file_dir):
            item_path = os.path.join(filter_mask_file_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pt'):
                filter_mask_file_paths.append(item_path)
        filter_mask_file_paths.sort()
        file_paths['filter_mask'] += filter_mask_file_paths
        
        return file_paths

class PresaveActivationDataset(ActivationDataset):
    def __init__(self, cfg: LorsaTrainConfig):
        presave_dataset = PresaveLoadingDataset(cfg)
        dataloader = DataLoader(
            presave_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.cfg = cfg
        
        self.act_buffer = {
            'input': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, self.cfg.lorsa_config.d_model, dtype=self.cfg.lorsa_config.dtype, device=self.cfg.lorsa_config.device),
            'output': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, self.cfg.lorsa_config.d_model, dtype=self.cfg.lorsa_config.dtype, device=self.cfg.lorsa_config.device),
            'filter_mask': torch.empty(self.cfg.buffer_size, self.cfg.lorsa_config.n_ctx, dtype=torch.bool, device=self.cfg.lorsa_config.device)
        }
        self.index = 0
        self.fill()

    def fill(self):
        while self.cfg.buffer_size - self.index >= self.cfg.lm_batch_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)
            input_tensor = batch['input'].to(self.cfg.lorsa_config.dtype).to(self.cfg.lorsa_config.device)
            output_tensor = batch['output'].to(self.cfg.lorsa_config.dtype).to(self.cfg.lorsa_config.device)
            filter_mask = batch['filter_mask'].to(self.cfg.lorsa_config.device)
            self.act_buffer['input'][self.index:self.index+self.cfg.lm_batch_size] = input_tensor
            self.act_buffer['output'][self.index:self.index+self.cfg.lm_batch_size] = output_tensor
            self.act_buffer['filter_mask'][self.index:self.index+self.cfg.lm_batch_size] = filter_mask
            self.index += self.cfg.lm_batch_size