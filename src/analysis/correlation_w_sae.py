"""lm_saes is required to run these analyses."""
import os
import torch
from einops import rearrange

from typing import Dict, Optional, Tuple, Union, List
from jaxtyping import Float, Int

from lm_saes import SparseAutoEncoder
from models.lorsa import LowRankSparseAttention
from transformer_lens import HookedTransformer
from datasets import Dataset
from tqdm.notebook import tqdm

@torch.no_grad()
def wo_lxa_sae_weight_based_correlation(
    sae: SparseAutoEncoder,
    lorsa: LowRankSparseAttention,
    top_n_sae_features: int = 8,
):
    lorsa.fold_W_O_into_W_V()  
    # saes are already transformed during loading from pretrained

    lorsa_wo = lorsa.W_O.data.squeeze(dim=1)  # n_ov_heads d_model, this should be unit normed along the d_model dim
    sae_enc = sae.encoder.weight.data  # d_sae d_model
    sae_dec = sae.decoder.weight.data  # d_model d_sae, this should be unit normed along the d_model dim

    assert sae_dec.norm(p=2, dim=0).isclose(torch.ones_like(sae_dec)).all()

    # multiplying lorsa WO and sae decoder immediately gives a cos sim matrix
    lorsa_wo_sae_dec_cos_sim = lorsa_wo @ sae_dec  # n_ov_heads d_sae
    lorsa_wo_sae_dec_cos_sim = lorsa_wo_sae_dec_cos_sim.sort(dim=1, descending=True)

    most_correlated_sae_features = lorsa_wo_sae_dec_cos_sim.indices[:, :top_n_sae_features]  # n_ov_heads top_n_sae_features
    most_correlated_sae_feature_cos_sims = lorsa_wo_sae_dec_cos_sim.values[:, :top_n_sae_features]  # n_ov_heads top_n_sae_features

    most_anti_correlated_sae_features = lorsa_wo_sae_dec_cos_sim.indices[:, -top_n_sae_features:]  # n_ov_heads top_n_sae_features
    most_anti_correlated_sae_feature_cos_sims = lorsa_wo_sae_dec_cos_sim.values[:, -top_n_sae_features:]  # n_ov_heads top_n_sae_features

    lorsa_wo_sae_enc_dfa = lorsa_wo @ sae_enc.T  # n_ov_heads d_sae
    most_correlated_feature_dfas = torch.gather(
        input=lorsa_wo_sae_enc_dfa,
        dim=1,
        index=most_correlated_sae_features,
    )  # n_ov_heads top_n_sae_features

    most_anti_correlated_feature_dfas = torch.gather(
        input=lorsa_wo_sae_enc_dfa,
        dim=1,
        index=most_anti_correlated_sae_features,
    )  # n_ov_heads top_n_sae_features

    return {
        'wo_lxa_most_correlated_sae_features': most_correlated_sae_features,
        'wo_lxa_most_correlated_sae_feature_cos_sims': most_correlated_sae_feature_cos_sims,
        'wo_lxa_most_correlated_feature_dfas': most_correlated_feature_dfas,
        'wo_lxa_most_anti_correlated_sae_features': most_anti_correlated_sae_features,
        'wo_lxa_most_anti_correlated_sae_feature_cos_sims': most_anti_correlated_sae_feature_cos_sims,
        'wo_lxa_most_anti_correlated_feature_dfas': most_anti_correlated_feature_dfas,
    }

@torch.no_grad()
def wv_lxain_sae_weight_based_correlation(
    sae: SparseAutoEncoder,
    lorsa: LowRankSparseAttention,
    top_n_sae_features: int = 8,
):
    """wv is no longer guaranteed to have unit norm."""

    lorsa.fold_W_O_into_W_V()  
    # saes are already transformed during loading from pretrained

    lorsa_wv = lorsa.W_V.data.squeeze(dim=2)  # n_ov_heads d_model

    sae_enc = sae.encoder.weight.data  # d_sae d_model
    sae_dec = sae.decoder.weight.data  # d_model d_sae, this should be unit normed along the d_model dim

    assert sae_dec.norm(p=2, dim=0).isclose(torch.ones_like(sae_dec)).all()   

    # direct feature attribution, no unit norm assumptions for lorsa wv
    lorsa_wv_sae_dec_dfa = lorsa_wv @ sae_dec  # n_ov_heads d_sae
    lorsa_wv_sae_dec_dfa = lorsa_wv_sae_dec_dfa.sort(dim=1, descending=True)

    most_correlated_sae_features = lorsa_wv_sae_dec_dfa.indices[:, :top_n_sae_features]  # n_ov_heads top_n_sae_features
    most_correlated_sae_feature_dfa = lorsa_wv_sae_dec_dfa.values[:, :top_n_sae_features]  # n_ov_heads top_n_sae_features

    most_anti_correlated_sae_features = lorsa_wv_sae_dec_dfa.indices[:, -top_n_sae_features:]  # n_ov_heads top_n_sae_features
    most_anti_correlated_sae_feature_dfa = lorsa_wv_sae_dec_dfa.values[:, -top_n_sae_features:]  # n_ov_heads top_n_sae_features

    return {
        'wv_lxain_most_correlated_sae_features': most_correlated_sae_features,
        'wv_lxain_most_correlated_sae_features_dfas': most_correlated_sae_feature_dfa,
        'wv_lxain_most_correlated_sae_features': most_anti_correlated_sae_features,
        'wv_lxain_most_correlated_sae_features_dfas': most_anti_correlated_sae_feature_dfa,
    }


@torch.no_grad()
def correlation_via_sample_results(
    sae: SparseAutoEncoder,
    model: HookedTransformer,
    most_interested_contexts: List[str],
    max_seq_len: int,
    q_positons: Int[torch.Tensor, "len(most_interested_contexts)"],
    k_positons: Int[torch.Tensor, "len(most_interested_contexts)"] | None = None,
    top_n_sae_features: int = 8,
):
    is_in_output_space = 'hook_attn_out' in sae.cfg.hook_point_in

    assert len(q_positons.size()) == 1
    assert q_positons.size(0) == len(most_interested_contexts)

    if not is_in_output_space:
        assert q_positons.size(0) == k_positons.size(0)
        assert len(k_positons.size()) == 1

    tokens = model.to_tokens(most_interested_contexts)[:, :max_seq_len]
    _, cache = model.run_with_cache_until(tokens, names_filter=[sae.cfg.hook_point_in])

    activations = cache[sae.cfg.hook_point_in]  # batch_size seq_len 
    feature_acts = sae.encode(activations)  # batch_size seq_len d_sae

    def gather_specific_pos_act(
        feature_acts: Float[torch.Tensor, "batch_size seq_len d_sae"],
        indices: Int[torch.Tensor, "batch_size"],
    ) -> Float[torch.Tensor, "batch_size d_sae"]:
        d_sae = feature_acts.size(-1)
        indices_expanded = indices.view(-1, 1, 1).expand(-1, 1, d_sae)
        return torch.gather(feature_acts, dim=1, index=indices_expanded).squeeze(1)

    q_pos_feature_act = gather_specific_pos_act(feature_acts, q_positons).mean(dim=0).sort(descending=True)  # d_sae

    results = {
        f'sample_based_max_correlated_{"OVoutput" if is_in_output_space else "q_pos"}_features': q_pos_feature_act.indices[:top_n_sae_features],  # top_n_sae_features
        f'sample_based_max_correlated_{"OVoutput" if is_in_output_space else "q_pos"}_feature_acts': q_pos_feature_act.values[:top_n_sae_features],  # top_n_sae_features
    }

    if not is_in_output_space:
        k_pos_feature_act = gather_specific_pos_act(feature_acts, k_positons).mean(dim=0).sort(descending=True)  # d_sae
        results['sample_based_max_correlated_k_pos_features'] = k_pos_feature_act.indices[:top_n_sae_features]  # top_n_sae_features
        results['sample_based_max_correlated_k_pos_feature_acts'] = k_pos_feature_act.values[:top_n_sae_features]  # top_n_sae_features
    
    return results


@torch.no_grad()
def correlation_via_sample_results_wrapper(
    lorsa: LowRankSparseAttention,
    lxa_sae: SparseAutoEncoder,
    lxain_sae: SparseAutoEncoder,
    model: HookedTransformer,
    dataset: Dataset,
    lorsa_sample_results: Dict[str, torch.Tensor],
    top_n_sae_features: int = 8,
):
    results = {
        'sample_based_max_correlated_OVoutput_features': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=torch.int32,
            device=lorsa.cfg.device,
        ),
        'sample_based_max_correlated_OVoutput_feature_acts': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=lorsa.cfg.dtype,
            device=lorsa.cfg.device,
        ),
        'sample_based_max_correlated_q_pos_features': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=torch.int32,
            device=lorsa.cfg.device,
        ),
        'sample_based_max_correlated_q_pos_feature_acts': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=lorsa.cfg.dtype,
            device=lorsa.cfg.device,
        ),
        'sample_based_max_correlated_k_pos_features': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=torch.int32,
            device=lorsa.cfg.device,
        ),
        'sample_based_max_correlated_k_pos_feature_acts': torch.empty(
            (lorsa.cfg.n_ov_heads, top_n_sae_features),
            dtype=lorsa.cfg.dtype,
            device=lorsa.cfg.device,
        ),
    }

    for i in tqdm(range(lorsa.cfg.n_ov_heads), desc="sample_based_correlation"):
        most_interested_contexts = dataset.select(
            lorsa_sample_results['context_idx'][i].cpu().numpy().tolist()
        )['text']

        qk_sample_based_corr_of_head_i = correlation_via_sample_results(
            sae=lxain_sae,
            model=model,
            most_interested_contexts=most_interested_contexts,
            max_seq_len=lorsa.cfg.n_ctx,
            q_positons=lorsa_sample_results['q_pos_of_max_activating_samples'][i],
            k_positons=lorsa_sample_results['dfa_of_max_activating_samples'][i].max(dim=-1).indices,
            top_n_sae_features=top_n_sae_features,
        )

        o_sample_based_corr_of_head_i = correlation_via_sample_results(
            sae=lxa_sae,
            model=model,
            most_interested_contexts=most_interested_contexts,
            max_seq_len=lorsa.cfg.n_ctx,
            q_positons=lorsa_sample_results['q_pos_of_max_activating_samples'][i],
            k_positons=None,
            top_n_sae_features=top_n_sae_features,
        )

        for key in qk_sample_based_corr_of_head_i:
            results[key][i] = qk_sample_based_corr_of_head_i[key]
        
        for key in o_sample_based_corr_of_head_i:
            results[key][i] = o_sample_based_corr_of_head_i[key]
    
    return results



@torch.no_grad()
def wqk_lxain_sae_weight_based_correlation(
    sae: SparseAutoEncoder,
    lorsa: LowRankSparseAttention,
    q_interested_features: Int[torch.Tensor, "n_ov_heads top_n_sae_features"],
    k_interested_features: Int[torch.Tensor, "n_ov_heads top_n_sae_features"],
):
    lorsa_wq, lorsa_wk = lorsa.W_Q.data, lorsa.W_K.data  # n_qk_heads d_model d_qk_head
    sae_dec = sae.decoder.weight.data  # d_model d_sae, this should be unit normed along the d_model dim

    assert sae_dec.norm(p=2, dim=0).isclose(torch.ones_like(sae_dec)).all()
    assert q_interested_features.size() == k_interested_features.size()

    n_ov_heads = lorsa.cfg.n_ov_heads
    n_qk_heads = lorsa_wq.size(0)
    d_sae = sae_dec.size(1)
    top_n_sae_features = q_interested_features.size(-1)
    n_bind = n_ov_heads // n_qk_heads

    assert n_ov_heads % n_qk_heads == 0

    # U: n_qk_heads d_model d_qk_head, we naturally assume d_model > d_qk_head
    # S: n_qk_heads d_qk_head
    # V: n_qk_heads d_qk_head d_model
    U_q, S_q, Vh_q = torch.linalg.svd(lorsa_wq, full_matrices=False)  
    U_k, S_k, Vh_k = torch.linalg.svd(lorsa_wk, full_matrices=False)

    tolerance = 1e-8

    q_ranks = (S_q > tolerance).sum(dim=-1)  # n_qk_heads
    k_ranks = (S_k > tolerance).sum(dim=-1)  # n_qk_heads    

    q_space_fraction_of_sae_dec_norms = torch.zeros(
        (n_ov_heads, top_n_sae_features), 
        dtype=lorsa_wq.dtype, 
        device=lorsa_wq.device
    )
    k_space_fraction_of_sae_dec_norms = torch.zeros(
        (n_ov_heads, top_n_sae_features), 
        dtype=lorsa_wq.dtype, 
        device=lorsa_wq.device
    )

    q_interested_features = q_interested_features.view(n_qk_heads, n_bind, top_n_sae_features)  # n_qk_heads n_bind top_n_sae_features
    k_interested_features = k_interested_features.view(n_qk_heads, n_bind, top_n_sae_features)  # n_qk_heads n_bind top_n_sae_features


    for i in tqdm(range(n_qk_heads), desc="qk_weight_based_correlation"):
        q_column_space = U_q[i, :, :q_ranks[i]]  # d_model rank
        k_column_space = U_k[i, :, :k_ranks[i]]  # d_model rank

        # project LXAin decoder rows onto this space, this gives how much of each feature's decoder norm lies in this space
        # this is not quite precise in that it does not consider the singular values, i.e., 
        # we are missing in structures of our WQ/WK spaces, nor it takes into account qk interactions.
        # this should be viewed as a loose approximation of sae decoder & W_Q / W_K.
        q_percentile_in_column_spaces = q_column_space @ q_column_space.T @ sae_dec  # d_model d_sae
        q_percentile_in_column_spaces = q_percentile_in_column_spaces.norm(dim=0)  # d_sae

        k_percentile_in_column_spaces = k_column_space @ k_column_space.T @ sae_dec  # d_model d_sae
        k_percentile_in_column_spaces = k_percentile_in_column_spaces.norm(dim=0)  # d_sae

        q_space_fraction_of_sae_dec_norms[i * n_bind: (i + 1) * n_bind] = q_percentile_in_column_spaces[q_interested_features[i]]
        k_space_fraction_of_sae_dec_norms[i * n_bind: (i + 1) * n_bind] = k_percentile_in_column_spaces[k_interested_features[i]]
    
    return {
        'wq_lxain_fraction_of_sae_dec_norms_covered_by_wq': q_space_fraction_of_sae_dec_norms,
        'wk_lxain_fraction_of_sae_dec_norms_covered_by_wk': k_space_fraction_of_sae_dec_norms,
    }


@torch.no_grad()
def correlation_analyses_between_saes_and_lorsas(
    lorsa: LowRankSparseAttention,
    lxa_sae: SparseAutoEncoder,
    lxain_sae: SparseAutoEncoder,
    model: HookedTransformer,
    dataset: Dataset,
    lorsa_sample_results: Dict[str, torch.Tensor],
    top_n_sae_features: int = 8,
):
    results = {}

    results.update(
        wo_lxa_sae_weight_based_correlation(
            sae=lxa_sae,
            lorsa=lorsa,
            top_n_sae_features=top_n_sae_features,
        )
    )

    results.update(
        wv_lxain_sae_weight_based_correlation(
            sae=lxain_sae,
            lorsa=lorsa,
            top_n_sae_features=top_n_sae_features,
        )
    )

    results.update(
        correlation_via_sample_results_wrapper(
            lorsa=lorsa,
            lxa_sae=lxa_sae,
            lxain_sae=lxain_sae,
            model=model,
            dataset=dataset,
            lorsa_sample_results=lorsa_sample_results,
            top_n_sae_features=top_n_sae_features,
        )
    )

    results.update(
        wqk_lxain_sae_weight_based_correlation(
            sae=lxain_sae,
            lorsa=lorsa,
            q_interested_features=results['sample_based_max_correlated_q_pos_features'],
            k_interested_features=results['sample_based_max_correlated_k_pos_features'],
        )
    )

    return results

