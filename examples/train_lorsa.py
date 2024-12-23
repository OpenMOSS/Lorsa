import sys
sys.path.append('src')

import argparse

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from config import LorsaTrainConfig, LorsaConfig
from runner import train_lorsa_runner

parser = argparse.ArgumentParser(description='Process hyparameters')
# orig attn config
parser.add_argument('--model_name', type=str, required=True, default="EleutherAI/pythia-160m", help='model name or model path')
parser.add_argument('-m', '--model', type=str, required=False, default="EleutherAI/pythia-160m", help='model name or model path')
parser.add_argument('-l', '--layer', type=int, required=False, default=3, help='Layer number')
parser.add_argument('--prepend_bos', action='store_true', help="Use prepend bos or not")
# dataset config
parser.add_argument('--dataset_path', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', choices=['text', 'activation'], required=True, help="Specify the mode. Must be 'text' or 'activation'.")
parser.add_argument('--num_workers', type=int, required=False, default=32, help='Num workers, default 32')
parser.add_argument('--lm_batch_size', type=int, required=False, default=32, help='Batchsize when forward, default 32')
parser.add_argument('--buffer_size', type=int, required=False, default=512, help='num of batchs in buffer')

# training config
parser.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Batchsize, default 32')
parser.add_argument('--total_tokens', type=int, required=False, default=2_000_000_000, help='Total tokens, default 2_000_000_000')
parser.add_argument('--lr', type=float, required=False, default=4e-4, help='Learning rate, default 4e-4')
parser.add_argument('--final_lr', type=float, required=False, default=4e-4, help='Final learning rate, default 4e-4')
parser.add_argument('--lr_warm_up_tokens', type=int, required=False, default=10_000_000, help='Total tokens, default 2_000_000_000')
parser.add_argument('--lr_cool_down_tokens', type=int, required=False, default=10_000_000, help='Total tokens, default 2_000_000_000')
parser.add_argument('--clip_grad_norm', type=float, required=False, default=1, help='max gradient norm')
parser.add_argument('--init_scale', action='store_true', help='scale parameters when init')
parser.add_argument('--mode', type=str, required=False, default='top_k', help='mode')
# k config
parser.add_argument('-k', '--top_k', type=int, required=False, default=16, help='nums of top heads to be used')
parser.add_argument('--start_k', type=int, required=False, default=16, help='initial nums of top heads')
parser.add_argument('--k_warm_up_tokens', type=int, required=False, default=0, help='k warm up tokens')
parser.add_argument('--k_scheduler_name', type=str, required=False, default='linear', help='k scheduler name')
# l1 config
parser.add_argument('--l1_coef', type=float, required=False, default=1e-5, help='l1 coef')
# wandb config
parser.add_argument('--log_to_wandb', action='store_true', help='log to wandb')
parser.add_argument('--wandb_project', type=str, required=False, default='pythia-160m-lorsa', help='wandb project')
parser.add_argument('--wandb_entity', type=str, required=False, default='fnlp-mechinterp', help='wandb entity')
# result config
parser.add_argument('--result_dir', type=str, required=False, default='./', help='result dir')
# lorsa config
parser.add_argument('--n_ctx', type=int, required=False, default=256, help='contexts length')
parser.add_argument('--n_qk_heads', type=int, required=False, default=128, help='nums of heads')
parser.add_argument('--n_ov_heads', type=int, required=False, default=512, help='nums of heads')
parser.add_argument('--d_qk_head', type=int, required=False, default=8, help='dim of qk head')
parser.add_argument('--d_ov_head', type=int, required=False, default=1, help='dim of ov head')
parser.add_argument('--rotary_scale', type=int, required=False, default=1, help='rotary scale')
parser.add_argument('--rotary_dim', type=int, required=False, default=8, help='rotary dim')
parser.add_argument('--use_z_relu', action='store_true', help='use relu on z')

args = parser.parse_args()

def main():
    if args.mode == "l1":
        project_name = f'L{args.layer}A-d{args.d_qk_head}&{args.d_ov_head}-n{args.n_qk_heads}&{args.n_ov_heads}-ctx{args.n_ctx}-lr{args.lr}-l1coef{args.l1_coef}'
    elif args.mode == "top_k":
        project_name = f'L{args.layer}A-d{args.d_qk_head}&{args.d_ov_head}-n{args.n_qk_heads}&{args.n_ov_heads}-ctx{args.n_ctx}-lr{args.lr}-k{args.top_k}'
    elif args.mode == "default":
        project_name = f'L{args.layer}A-d{args.d_qk_head}&{args.d_ov_head}-n{args.n_qk_heads}&{args.n_ov_heads}-ctx{args.n_ctx}-lr{args.lr}'

    if args.use_z_relu:
        project_name += '-relu'

    cfg = LorsaTrainConfig(
        # orig attention head config
        model_name = args.model_name,
        model = args.model,
        layer = args.layer,
        prepend_bos = args.prepend_bos,

        # dataset config
        dataset_path = args.dataset_path,
        dataset_type = args.dataset_type,
        num_workers = args.num_workers,
        lm_batch_size = args.lm_batch_size,
        buffer_size = args.buffer_size,

        # training config
        batch_size = args.batch_size,
        total_tokens = args.total_tokens,
        learning_rate = args.lr,
        final_learning_rate = args.final_lr,
        lr_warm_up_tokens = args.lr_warm_up_tokens,
        lr_cool_down_tokens = args.lr_cool_down_tokens,
        clip_grad_norm = args.clip_grad_norm,
        mode = args.mode,
        init_scale_parameters = args.init_scale,
        
        # k config
        k_scheduler_name = args.k_scheduler_name, # ['linear', 'exponential', 'cosine', 'smooth_step', 'sqrt']
        start_k = args.start_k,
        end_k = args.top_k,
        k_warm_up_tokens = args.k_warm_up_tokens,
        
        # l1 config
        l1_coef=args.l1_coef,

        # wandb config
        log_to_wandb = args.log_to_wandb,
        log_frequency = 100,
        wandb_project = args.wandb_project,
        wandb_entity = args.wandb_entity,
        project_name=project_name,
        
        # result config
        result_dir = args.result_dir,
        
        # lorsa config
        lorsa_config = LorsaConfig(
            # self attention head config
            n_ctx=args.n_ctx,
            d_qk_head = args.d_qk_head,
            d_ov_head = args.d_ov_head,
            n_qk_heads = args.n_qk_heads,
            n_ov_heads = args.n_ov_heads,
            rotary_scale = args.rotary_scale,
            rotary_dim=args.rotary_dim,
            use_z_relu=args.use_z_relu,
            top_k = args.top_k,
            device = "cuda",
            dtype = torch.float32,
        ),
    )

    train_lorsa_runner(cfg=cfg)
    
if __name__ == '__main__':
    main()