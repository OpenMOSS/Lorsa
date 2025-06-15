import torch

def estimate_memory_usage(batch_size, n_ctx, n_ov_heads, d_ov_head=1, dtype=torch.float32):
    """
    估算JumpReLU前向传播的内存使用
    """
    # tensor shape: (batch_size, n_ctx, n_ov_heads, d_ov_head)
    tensor_shape = (batch_size, n_ctx, n_ov_heads, d_ov_head)
    
    # 计算单个tensor的字节数
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        bytes_per_element = 2
    else:
        bytes_per_element = 4
    
    tensor_size_bytes = batch_size * n_ctx * n_ov_heads * d_ov_head * bytes_per_element
    tensor_size_gb = tensor_size_bytes / (1024**3)
    
    print(f"Tensor shape: {tensor_shape}")
    print(f"Single tensor size: {tensor_size_gb:.2f} GB")
    
    # JumpReLU中需要的tensor:
    # 1. x (input): tensor_size_gb
    # 2. threshold (broadcasted): ~0 (很小)
    # 3. mask: tensor_size_gb (boolean, 1 byte per element)
    # 4. output: tensor_size_gb
    # 5. 在原始实现中: torch.zeros_like(x): tensor_size_gb (额外的!)
    
    mask_size_gb = batch_size * n_ctx * n_ov_heads * d_ov_head / (1024**3)  # boolean is 1 byte
    
    print(f"\nMemory usage breakdown:")
    print(f"Input tensor (x): {tensor_size_gb:.2f} GB")
    print(f"Mask tensor: {mask_size_gb:.2f} GB")
    print(f"Output tensor: {tensor_size_gb:.2f} GB")
    
    # 原始实现
    original_memory = tensor_size_gb * 3 + mask_size_gb  # x + zeros_like + output + mask
    print(f"Original JumpReLU implementation: {original_memory:.2f} GB")
    
    # 优化后实现
    optimized_memory = tensor_size_gb * 2 + mask_size_gb  # x + output + mask (no zeros_like)
    print(f"Optimized JumpReLU implementation: {optimized_memory:.2f} GB")
    
    print(f"Memory saved: {original_memory - optimized_memory:.2f} GB")
    
    return tensor_size_gb, original_memory, optimized_memory

if __name__ == "__main__":
    print("=== Memory Analysis for JumpReLU ===\n")
    
    # 测试不同的参数配置
    test_configs = [
        (32, 256, 512, 1),   # 可能的配置
        (64, 256, 512, 1),   # 较大batch_size
        (32, 256, 1024, 1),  # 更多heads
        (32, 256, 2048, 1),  # 非常多heads
    ]
    
    for i, (batch_size, n_ctx, n_ov_heads, d_ov_head) in enumerate(test_configs):
        print(f"Config {i+1}: batch_size={batch_size}, n_ctx={n_ctx}, n_ov_heads={n_ov_heads}")
        estimate_memory_usage(batch_size, n_ctx, n_ov_heads, d_ov_head)
        print("-" * 60) 