import math

from models.attention import LowRankSparseAttention

class LrWarmupScheduler:
    def __init__(self, optimizer, base_lr, final_lr, warm_up_tokens, cool_down_tokens, total_tokens):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warm_up_tokens = warm_up_tokens
        self.cool_down_tokens = cool_down_tokens
        self.total_tokens = total_tokens
        self.current_tokens = 0

    def update_lr(self, current_tokens):
        self.current_tokens = current_tokens

        if self.current_tokens < self.warm_up_tokens:
            # Linear warm up phase
            lr = self.base_lr * (self.current_tokens / self.warm_up_tokens)
        elif self.current_tokens < self.total_tokens - self.cool_down_tokens:
            lr = self.base_lr
        else:
            lr = self.final_lr + (self.base_lr - self.final_lr) * (self.total_tokens - self.current_tokens) / self.cool_down_tokens

        # Apply the calculated learning rate to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
class TopkWarmupScheduler:
    def __init__(self, lorsa: LowRankSparseAttention, start_k, end_k, k_scheduler_name, warm_up_tokens, total_tokens):
        self.lorsa = lorsa
        self.start_k = start_k
        self.end_k = end_k
        self.k_scheduler_name = k_scheduler_name
        self.warm_up_tokens = warm_up_tokens
        self.total_tokens = total_tokens
        self.current_tokens = 0
        
    def update_k(self, current_tokens):
        self.current_tokens = current_tokens

        if self.current_tokens < self.warm_up_tokens:
            if self.k_scheduler_name == 'linear':
                k = self.start_k - (self.start_k - self.end_k) * (self.current_tokens / self.warm_up_tokens)
                
            elif self.k_scheduler_name == 'exponential':
                k = self.start_k * (self.end_k / self.start_k) ** (self.current_tokens / self.warm_up_tokens)
                
            elif self.k_scheduler_name == 'cosine':
                k = self.end_k + 0.5 * (self.start_k - self.end_k) * (1 + math.cos(math.pi * self.current_tokens / self.warm_up_tokens))
                
            elif self.k_scheduler_name == 'smooth_step':
                progress = self.current_tokens / self.warm_up_tokens
                k = self.start_k - (self.start_k - self.end_k) * (3 * progress**2 - 2 * progress**3)\
                    
            elif self.k_scheduler_name == 'sqrt':
                progress = self.current_tokens / self.warm_up_tokens
                k = self.start_k - (self.start_k - self.end_k) * math.sqrt(progress)
                
            # default linear
            else:
                k = self.start_k - (self.start_k - self.end_k) * (self.current_tokens / self.warm_up_tokens)
        else:
            k = self.end_k

        self.lorsa.cfg.top_k = int(k)
            
    def get_lr(self):
        return self.lorsa.cfg.top_k