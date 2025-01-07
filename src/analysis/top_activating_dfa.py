import os
import torch
from typing import List, Set, Tuple, Dict, Union
from jaxtyping import Float, Int, Bool
from models.lorsa import LowRankSparseAttention
from transformer_lens import HookedTransformer
from config import LorsaConfig
from tqdm.notebook import tqdm
from einops import rearrange, repeat
from utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor

@torch.no_grad()
def get_tokenized_text(
    lorsa: LowRankSparseAttention, 
    model: HookedTransformer, 
    text: List[str]
) -> List[List[str]]:
    tokens = model.to_str_tokens(text)
    return [t[:lorsa.cfg.n_ctx] for t in tokens]

@torch.no_grad()
def get_activation_with_filter_mask(
    model: HookedTransformer,
    batch: List[str],
    ignore_tokens: Set[int],
    cfg: LorsaConfig,
) -> Tuple[
    Float[torch.Tensor, "batch_size ctx_length d_model"],
    Bool[torch.Tensor, "batch_size ctx_length"],
]:
    tokens = model.to_tokens(
        batch, 
        prepend_bos=True,
    ).to(cfg.device, non_blocking=True)

    tokens = tokens[:, :cfg.n_ctx]

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

    return hook_in, filter_mask

@torch.no_grad()
def get_z_of_all_heads(
    lorsa: LowRankSparseAttention,
    activation: Float[torch.Tensor, "batch_size ctx_length d_model"],
    get_dfa: bool = False,
    interested_head_mask: Float[torch.Tensor, "n_ov_heads topn_activating_samples"] | None = None,
) -> Union[
    Float[torch.Tensor, "n_ov_heads batch_size ctx_length"],
    Tuple[
        Float[torch.Tensor, "n_ov_heads batch_size ctx_length"], 
        Float[torch.Tensor, "n_ov_heads batch_size"],
    ]
]:
    if get_dfa:
        batch_size, ctx_length, n_ov_heads = activation.size(0), activation.size(1), interested_head_mask.size(0)
        interested_head_mask = interested_head_mask.any(dim=1).nonzero().squeeze(1)  # reduced_n_ov_heads

        q, k, v, pattern = lorsa.cal_q_k_v_pattern(activation)

        z = lorsa.cal_per_key_position_z_with_h(
            v, 
            pattern, 
            interested_head_mask=interested_head_mask
        )  # batch_size query_pos key_pos reduced_n_ov_heads d_head

        z = z.squeeze(-1)  # batch_size query_pos key_pos reduced_n_ov_heads, we only care about 1-d lorsa heads now
        # place n_heads dimension to the first dim for per-head analysis
        z = z.permute(3, 0, 1, 2)  # reduced_n_ov_heads batch_size query_pos key_pos

        # sum over all keys, get the max activating q pos, getting a [reduced_n_ov_heads batch_size] index
        interested_query_pos = z.sum(dim=-1).max(dim=-1).indices  
        interested_query_pos_to_gather = interested_query_pos[:, :, None, None].expand(-1, -1, -1, z.size(3))
        z = z.gather(dim=2, index=interested_query_pos_to_gather).squeeze(2)  # DFA: reduced_n_ov_heads batch_size key_pos

        result_z = torch.zeros(
            (n_ov_heads, batch_size, ctx_length,),
            dtype=lorsa.cfg.dtype,
            device=activation.device,
        )

        result_interested_query_pos = torch.zeros(
            (n_ov_heads, batch_size,),
            dtype=torch.long,
            device=activation.device,
        )

        result_z[interested_head_mask] = z
        result_interested_query_pos[interested_head_mask] = interested_query_pos

        return result_z, result_interested_query_pos

        # This gives us the same size as when get_dfa=False, but the meaning of the 'z's are different.
        # This branch gives the DFA of the max activating query pos while the other gives us the
        # z of all query positions in the context of the max activating sample.
        
    else:
        _, z, _ = lorsa.cal_out_top_k_for_ov1(activation)  # batch_size ctx_length n_heads, d_head=1 has been squeezed in this method
        # place n_heads dimension to the first dim for per-head analysis
        z = z.permute(2, 0, 1)
    
        return z

@torch.no_grad()
def pad_tensor(tensor, dim, length):
    """
    Pads a tensor with zeros along the specified dimension until it reaches the given length.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension to pad.
        length (int): The target length for the specified dimension.
        
    Returns:
        torch.Tensor: The padded tensor.
    """
    # Get the current size of the tensor along each dimension
    size = list(tensor.size())
    
    # Check if padding is needed
    if size[dim] >= length:
        return tensor  # No padding needed
    
    # Compute the padding sizes
    pad_size = [0] * (2 * tensor.dim())  # Pad size for all dimensions (start and end for each dim)
    pad_size[-(2 * dim + 1)] = length - size[dim]  # Padding for the specified dimension (end)
    
    # Apply padding
    return torch.nn.functional.pad(tensor, pad=pad_size)


@torch.no_grad()
def get_dfa_of_max_activating_samples(
    lorsa: LowRankSparseAttention,
    dataset: torch.utils.data.Dataset,
    model: HookedTransformer,
    ignore_tokens: Set[int],
    sample_results: Dict[str, torch.Tensor],
    get_topn_activating_samples: int
):
    """
    We do not actually save the top activating samples or DFA result while iterating
    the analysis dataset to save GPU memory. Instead we record indices of the max 
    activating dataset samples and rerun the key results we need here.
    """
    interested_indexes = torch.unique(
        sample_results['context_idx'].flatten()[sample_results['elt'].ne(0).flatten()],
        sorted=False,
    )
    dfa_of_max_activating_samples = torch.zeros(
        (lorsa.cfg.n_ov_heads, get_topn_activating_samples, lorsa.cfg.n_ctx),
        dtype=lorsa.cfg.dtype,
        device=lorsa.cfg.device,
    )

    q_pos_of_max_activating_samples = torch.zeros(
        (lorsa.cfg.n_ov_heads, get_topn_activating_samples),
        dtype=torch.int32,
        device=lorsa.cfg.device,
    )

    interested_subset = dataset.select(interested_indexes.cpu().numpy().tolist())

    dataloader = torch.utils.data.DataLoader(
        interested_subset['text'], 
        batch_size=1, 
        shuffle=False, 
        drop_last=False,
    )
    for i, batch in enumerate(tqdm(dataloader)):
        activation, filter_mask = get_activation_with_filter_mask(
            model=model,
            batch=batch,
            ignore_tokens=ignore_tokens,
            cfg=lorsa.cfg,
        )

        interested_mask = sample_results['context_idx'] == interested_indexes[i]  # n_heads topn_activating_samples
        interested_mask *= sample_results['elt'].ne(0)
        

        dfa, q_position = get_z_of_all_heads(
            lorsa, 
            activation, 
            get_dfa=True, 
            interested_head_mask=interested_mask,
        )  # n_heads 1 ctx_length ; n_heads 1

        
        dfa_of_max_activating_samples = torch.where(
            condition=interested_mask[:, :, None],  # n_heads topn_activating_samples 1
            input=pad_tensor(
                tensor=dfa,
                dim=2,
                length=lorsa.cfg.n_ctx,
            ),  # n_heads 1 ctx_length
            other=dfa_of_max_activating_samples,  # n_heads topn_activating_samples ctx_length
        )

        q_pos_of_max_activating_samples = torch.where(
            condition=interested_mask,  # n_heads topn_activating_samples
            input=q_position,
            other=q_pos_of_max_activating_samples,  # n_heads topn_activating_samples
        )

    return dfa_of_max_activating_samples, q_pos_of_max_activating_samples

    
@torch.no_grad()
def sample_max_activating_sequences(
    lorsa: LowRankSparseAttention,
    dataset: torch.utils.data.Dataset,
    model: HookedTransformer,
    ignore_tokens: Set[int],
    batch_size: int = 32,
    get_topn_activating_samples: int = 32,
):
    cfg = lorsa.cfg

    dataloader = torch.utils.data.DataLoader(
        dataset['text'], 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
    )

    sample_results = {
        "elt": torch.empty(
            (cfg.n_ov_heads, 0), 
            dtype=cfg.dtype, 
            device=cfg.device
        ),
        "context_idx": torch.empty(
            (cfg.n_ov_heads, 0),
            dtype=torch.int32,
            device=cfg.device,
        ),
    }

    act_times = torch.zeros(
        (cfg.n_ov_heads,), 
        dtype=torch.long, 
        device=cfg.device,
    )

    for i, batch in enumerate(tqdm(dataloader)):
        activation, filter_mask = get_activation_with_filter_mask(
            model=model,
            batch=batch,
            ignore_tokens=ignore_tokens,
            cfg=cfg,
        )

        z = get_z_of_all_heads(lorsa, activation)  # n_ov_heads batch_size ctx_length
        z = z.where(~filter_mask[None, ...], 0.0)  # shape remains, filter ignore tokens

        elt = z.max(dim=-1).values  # n_ov_heads batch_size

        act_times += z.gt(0.0).sum(dim=[1, 2])

        if (
            sample_results["elt"].size(1) > 0
            and (elt.max(dim=1).values <= sample_results["elt"][:, -1]).all()
        ):
            continue

        sample_results = concat_dict_of_tensor(
            sample_results,
            {
                "elt": elt,
                "context_idx": repeat(
                    torch.arange(
                        i * batch_size, 
                        (i + 1) * batch_size,
                        dtype=torch.int32,
                        device=cfg.device,
                    ),
                    "batch_size -> n_ov_heads batch_size",
                    n_ov_heads=cfg.n_ov_heads,
                )
            },
            dim=1,
        )

        sample_results = sort_dict_of_tensor(
            sample_results, 
            sort_dim=1, 
            sort_key="elt", 
            descending=True
        )

        sample_results = {
            k: v[:, :get_topn_activating_samples] 
            for k, v in sample_results.items()
        }
        
    dfa_of_max_activating_samples, q_pos_of_max_activating_samples = get_dfa_of_max_activating_samples(
        lorsa=lorsa,
        dataset=dataset,
        model=model,
        ignore_tokens=ignore_tokens,
        sample_results=sample_results,
        get_topn_activating_samples=get_topn_activating_samples,
    )

    sample_results['dfa_of_max_activating_samples'] = dfa_of_max_activating_samples
    sample_results['q_pos_of_max_activating_samples'] = q_pos_of_max_activating_samples
    sample_results['act_times'] = act_times
    
    return sample_results




        




        