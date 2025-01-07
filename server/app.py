import io
import os
from typing import Annotated, Any, Literal, Union, cast, List
from jaxtyping import Float
import random
import sys
sys.path.append('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Lorsa/src')

import msgpack
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from models.lorsa import LowRankSparseAttention
from utils.bytes import bytes_to_unicode
from config import LorsaConfig

result_dir = os.environ.get("RESULT_DIR", "results")
dataset_path = os.environ.get("DATASET_PATH", None)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

lm_cache = {}
dataset_cache: dict[str, Dataset] = {}


def get_model(lorsa_name: str) -> HookedTransformer:
    MODEL_PATH = {
        'EleutherAI/pythia-160m': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/pythia-160m',
        "meta-llama/Llama-3.1-8B": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/Llama-3.1-8B",
    }
    
    path = f"{result_dir}/{lorsa_name}"
    cfg = LorsaConfig.from_pretrained(path=path)
    
    model_path = MODEL_PATH[cfg.model_name]

    if (cfg.model_name, model_path) not in lm_cache:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=True,
        )

        model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, 
            use_flash_attn=True, 
            hf_model=hf_model,
            hf_config=hf_model.config,
            tokenizer=tokenizer,
            device=device,
            dtype=cfg.dtype,
        )
        lm_cache[(cfg.model_name, model_path)] = model
    return lm_cache[(cfg.model_name, model_path)]


def get_dataset() -> Dataset:
    dataset_cache[0] = load_from_disk(dataset_path)
    return dataset_cache[0]


def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    print("Current cache:", sae_cache.keys())
    # Clear cache
    sae_cache.clear()
    return Response(content="CUDA Out of memory", status_code=500)


def pack_text_and_dfa(
    text: List[str], 
    dfa: Float[torch.Tensor, "batch context_len"],
    qpos: Float[torch.Tensor, "batch"],
    model: HookedTransformer,
):
    result = []
    assert len(text) == dfa.size(0)
    max_context_len = dfa.size(1)
    for sentence, single_sentence_dfa, single_text_qpos in zip(text, dfa, qpos):
        tokens = model.to_str_tokens(sentence)[:max_context_len]
        actual_context_len = len(tokens)

        single_sentence_dfa = single_sentence_dfa[:actual_context_len]

        indices = (single_sentence_dfa == 0.).nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            q_position = indices[0].item()
        else:
            q_position = actual_context_len

        small_value_threshold = 0.001 * single_sentence_dfa.max().item()
        single_sentence_dfa = torch.where(
            single_sentence_dfa < small_value_threshold,
            0.,
            single_sentence_dfa
        )

        result.append(
            {'context': tokens, 'head_acts': single_sentence_dfa, 'q_position': single_text_qpos.item()}
        )

    return result


@app.get("/lorsas/{lorsa_name}/heads/{head_index}")
def get_head(lorsa_name: str, head_index: str | int):
    model = get_model(lorsa_name)
    if isinstance(head_index, str) and head_index != "random":
        try:
            head_index = int(head_index)
        except ValueError:
            return Response(
                content=f"Head index {head_index} is not a valid integer",
                status_code=400,
            )
    
    dataset = get_dataset()
    sample_results = torch.load(f"{result_dir}/{lorsa_name}/sample_results.pt", map_location=device)

    if head_index == "random":
        # head_index = random.randint(0, sample_results['elt'].size(0) - 1)
        head_index = random.choice(
            (sample_results['act_times'] > 1000).nonzero().squeeze(1).cpu().tolist()
        )

    return Response(
        content=msgpack.packb(
            make_serializable({
                'head_index': head_index,
                'lorsa_name': lorsa_name,
                'max_head_act': sample_results['elt'][head_index].max().item(),
                'context_idx': sample_results['context_idx'][head_index],
                'samples': pack_text_and_dfa(
                    text=dataset.select(
                        sample_results['context_idx'][head_index].cpu().numpy().tolist()
                    )['text'],
                    dfa=sample_results['dfa_of_max_activating_samples'][head_index],
                    qpos=sample_results['q_pos_of_max_activating_samples'][head_index],
                    model=model,
                ),
                'act_times': sample_results['act_times'][head_index].item(),
                # 'correlation_to_saes': {
                #     'wo_lxa': {
                #         'most_correlated_features_weight_based': {
                #             'feature_id': sample_results['wo_lxa_most_correlated_sae_features'][head_index],
                #             'decoder_cosine_similarities': sample_results['wo_lxa_most_correlated_sae_feature_cos_sims'][head_index],
                #             'wo_encoder_dfa': sample_results['wo_lxa_most_correlated_feature_dfas'][head_index],
                #         },
                #         'most_anti_correlated_features_weight_based': {
                #             'feature_id': sample_results['wo_lxa_most_anti_correlated_sae_features'][head_index],
                #             'decoder_cosine_similarities': sample_results['wo_lxa_most_anti_correlated_sae_feature_cos_sims'][head_index],
                #             'wo_encoder_dfa': sample_results['wo_lxa_most_anti_correlated_feature_dfas'][head_index],
                #         },
                #         'most_correlated_features_sample_based': {
                #             'feature_id': sample_results['sample_based_max_correlated_OVoutput_features'][head_index],
                #             'average_feature_act_given_lorsa_is_activated': sample_results['sample_based_max_correlated_OVoutput_feature_acts'][head_index],
                #         },
                #     },
                #     'wv_lxain': {
                #         'most_correlated_features_weight_based': {
                #             'feature_id': sample_results['wv_lxain_most_correlated_sae_features'][head_index],
                #             'decoder_wv_dfa': sample_results['wv_lxain_most_correlated_sae_features_dfas'][head_index],
                #         },
                #         'most_correlated_features_sample_based': {
                #             'feature_id': sample_results['sample_based_max_correlated_k_pos_features'][head_index],
                #             'average_feature_act_given_lorsa_is_activated': sample_results['sample_based_max_correlated_k_pos_feature_acts'][head_index],
                #         },
                #     },
                #     'wq_lxain': {
                #         'most_correlated_features_sample_based': {
                #             'feature_id': sample_results['sample_based_max_correlated_q_pos_features'][head_index],
                #             'average_feature_act_given_lorsa_is_activated': sample_results['sample_based_max_correlated_q_pos_feature_acts'][head_index],
                #             'fraction_of_norm_in_column_space': sample_results['wq_lxain_fraction_of_sae_dec_norms_covered_by_wq'][head_index],
                #         },
                #     },
                #     'wk_lxain': {
                #         'most_correlated_features_sample_based': {
                #             'feature_id': sample_results['sample_based_max_correlated_k_pos_features'][head_index],
                #             'average_feature_act_given_lorsa_is_activated': sample_results['sample_based_max_correlated_k_pos_feature_acts'][head_index],
                #             'fraction_of_norm_in_column_space': sample_results['wk_lxain_fraction_of_sae_dec_norms_covered_by_wk'][head_index],
                #         },
                #     },
                # },
                "interpretation": None,
            })
        ),
        media_type="application/x-msgpack",
    )


@app.get("/lorsas")
def list_lorsas():
    return os.listdir(result_dir)


@app.get("/dictionaries/{dictionary_name}")
def get_dictionary(dictionary_name: str):
    feature_activation_times = client.get_feature_act_times(dictionary_name, dictionary_series=dictionary_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    alive_feature_count = client.get_alive_feature_count(dictionary_name, dictionary_series=dictionary_series)
    if alive_feature_count is None:
        return Response(content=f"Dictionary {dictionary_name} not found", status_code=404)

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "dictionary_name": dictionary_name,
                    "feature_activation_times_histogram": [feature_activation_times_histogram],
                    "alive_feature_count": alive_feature_count,
                }
            )
        ),
        media_type="application/x-msgpack",
    )



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)