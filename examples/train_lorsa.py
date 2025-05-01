"""
Complete LoRSA Training Example

This script demonstrates all available configuration options for training a LoRSA model.
All parameters from the original implementation are preserved.
"""

import torch
import argparse
from config import LorsaTrainConfig, LorsaConfig
from runner import train_lorsa_runner

def parse_args():
    """Parse all available command line arguments for LoRSA training."""
    parser = argparse.ArgumentParser(description='Train LoRSA with full configuration')
    
    # ======================
    # Original Attention Config
    # ======================
    orig_attn_group = parser.add_argument_group('Original Attention Configuration')
    orig_attn_group.add_argument('--model_name', type=str, required=True,
                               help='Model name')
    orig_attn_group.add_argument('-m', '--model', type=str, default="EleutherAI/pythia-160m",
                               help='Model name or path')
    orig_attn_group.add_argument('-l', '--layer', type=int, default=3,
                               help='Layer number to intervene')
    orig_attn_group.add_argument('--prepend_bos', action='store_true',
                               help='Prepend BOS token to inputs')

    # ======================
    # Dataset Config
    # ======================
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset_path', type=str, required=True,
                             help='Path to training dataset')
    dataset_group.add_argument('--dataset_type', choices=['text', 'activation'], required=True,
                             help="Dataset type: 'text' or 'activation'")
    dataset_group.add_argument('--num_workers', type=int, default=32,
                             help='Number of data loader workers')
    dataset_group.add_argument('--lm_batch_size', type=int, default=32,
                             help='Language model forward batch size')
    dataset_group.add_argument('--buffer_size', type=int, default=512,
                             help='Number of batches in buffer')
    dataset_group.add_argument('--prefetch_factor', type=int, default=2,
                             help='Data loader prefetch factor')

    # ======================
    # Training Config
    # ======================
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('-b', '--batch_size', type=int, default=32,
                           help='Training batch size')
    train_group.add_argument('--total_tokens', type=int, default=2_000_000_000,
                           help='Total tokens to train on')
    train_group.add_argument('--lr', type=float, default=4e-4,
                           help='Initial learning rate')
    train_group.add_argument('--final_lr', type=float, default=4e-6,
                           help='Final learning rate')
    train_group.add_argument('--lr_warm_up_tokens', type=int, default=10_000_000,
                           help='Tokens for learning rate warmup')
    train_group.add_argument('--lr_cool_down_tokens', type=int, default=10_000_000,
                           help='Tokens for learning rate cooldown')
    train_group.add_argument('--clip_grad_norm', type=float, default=1.0,
                           help='Gradient clipping norm')
    train_group.add_argument('--init_scale', action='store_true',
                           help='Scale parameters during initialization')
    train_group.add_argument('--mode', type=str, default='top_k',
                           help='Training mode')

    # ======================
    # Top-k Config
    # ======================
    topk_group = parser.add_argument_group('Top-k Configuration')
    topk_group.add_argument('-k', '--top_k', type=int, default=16,
                          help='Final number of top heads to use')
    topk_group.add_argument('--start_k', type=int, default=16,
                          help='Initial number of top heads')
    topk_group.add_argument('--k_warm_up_tokens', type=int, default=0,
                          help='Tokens for k warmup')
    topk_group.add_argument('--k_scheduler_name', type=str, default='linear',
                          help='k scheduler type (linear/exponential/cosine/smooth_step/sqrt)')

    # ======================
    # L1 Regularization
    # ======================
    parser.add_argument('--l1_coef', type=float, default=1e-5,
                      help='L1 regularization coefficient')

    # ======================
    # WandB Config
    # ======================
    wandb_group = parser.add_argument_group('WandB Configuration')
    wandb_group.add_argument('--log_to_wandb', action='store_true',
                           help='Enable WandB logging')
    wandb_group.add_argument('--wandb_project', type=str, default='pythia-160m-lorsa',
                           help='WandB project name')
    wandb_group.add_argument('--wandb_entity', type=str, default='mechinterp',
                           help='WandB entity')
    wandb_group.add_argument('--project_name', type=str, default='LXA',
                           help='Project name for logging')

    # ======================
    # Output Config
    # ======================
    parser.add_argument('--result_dir', type=str, default='./result',
                      help='Directory for saving results')

    # ======================
    # LoRSA Model Config
    # ======================
    lorsa_group = parser.add_argument_group('LoRSA Model Configuration')
    lorsa_group.add_argument('--n_ctx', type=int, default=256,
                           help='Context length')
    lorsa_group.add_argument('--n_qk_heads', type=int, default=128,
                           help='Number of QK heads')
    lorsa_group.add_argument('--n_ov_heads', type=int, default=512,
                           help='Number of OV heads')
    lorsa_group.add_argument('--d_qk_head', type=int, default=8,
                           help='Dimension of each QK head')
    lorsa_group.add_argument('--d_ov_head', type=int, default=1,
                           help='Dimension of each OV head')
    lorsa_group.add_argument('--rotary_scale', type=int, default=1,
                           help='Rotary embedding scale factor')
    lorsa_group.add_argument('--rotary_dim', type=int, default=8,
                           help='Rotary embedding dimension')
    lorsa_group.add_argument('--use_z_relu', action='store_true',
                           help='Apply ReLU to z values')

    # ======================
    # Initialization Config
    # ======================
    init_group = parser.add_argument_group('Initialization Configuration')
    init_group.add_argument('--init_qk_with_orig_qk', action='store_true',
                          help='Initialize QK with original attention weights')
    init_group.add_argument('--fix_qk', action='store_true',
                          help='Fix QK parameters during training')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Special handling for activation datasets
    if args.dataset_type == 'activation':
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    # Build complete configuration
    cfg = LorsaTrainConfig(
        # Initialization
        init_qk_with_orig_qk=args.init_qk_with_orig_qk,
        fix_qk=args.fix_qk,
        
        # Original Attention
        model_name=args.model_name,
        model=args.model,
        layer=args.layer,
        prepend_bos=args.prepend_bos,

        # Dataset
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        num_workers=args.num_workers,
        lm_batch_size=args.lm_batch_size,
        buffer_size=args.buffer_size,
        prefetch_factor=args.prefetch_factor,

        # Training
        batch_size=args.batch_size,
        total_tokens=args.total_tokens,
        learning_rate=args.lr,
        final_learning_rate=args.final_lr,
        lr_warm_up_tokens=args.lr_warm_up_tokens,
        lr_cool_down_tokens=args.lr_cool_down_tokens,
        clip_grad_norm=args.clip_grad_norm,
        mode=args.mode,
        init_scale_parameters=args.init_scale,
        
        # Top-k
        k_scheduler_name=args.k_scheduler_name,
        start_k=args.start_k,
        end_k=args.top_k,
        k_warm_up_tokens=args.k_warm_up_tokens,
        
        # L1
        l1_coef=args.l1_coef,

        # WandB
        log_to_wandb=args.log_to_wandb,
        log_frequency=100,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        project_name=args.project_name,
        
        # Output
        result_dir=args.result_dir,
        
        # LoRSA Model
        lorsa_config=LorsaConfig(
            n_ctx=args.n_ctx,
            d_qk_head=args.d_qk_head,
            d_ov_head=args.d_ov_head,
            n_qk_heads=args.n_qk_heads,
            n_ov_heads=args.n_ov_heads,
            rotary_scale=args.rotary_scale,
            rotary_dim=args.rotary_dim,
            use_z_relu=args.use_z_relu,
            top_k=args.top_k,
            device="cuda",
            dtype=torch.float32,
        ),
    )

    # Start training
    train_lorsa_runner(cfg=cfg)

if __name__ == '__main__':
    main()