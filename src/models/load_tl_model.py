from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)


def load_tl_model(model_name: str):
    device = 'cuda'
    model_path = {
        "meta-llama/Llama-3.1-8B": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/Llama-3.1-8B",
        "EleutherAI/pythia-160m": "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/models/pythia-160m",
    }[model_name]
    
    dtype = {
        "meta-llama/Llama-3.1-8B": torch.bfloat16,
        "EleutherAI/pythia-160m": torch.float32,
    }[model_name]

    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
        local_files_only=True,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        use_flash_attn=False,
        device=device,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        processor=hf_processor,
        dtype=dtype,
        hf_config=hf_model.config,
    )
    model.eval()
    return model