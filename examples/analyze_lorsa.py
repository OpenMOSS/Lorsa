import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from models.lorsa import LowRankSparseAttention
from analysis.new_analysis import sample_max_activating_sequences
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Analyze LoRSA attention patterns')
parser.add_argument('--lorsa_path', type=str, required=True, 
                    help='Path to pretrained LoRSA model')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B", 
                    help='HuggingFace model name')
parser.add_argument('--model_path', type=str, 
                    help='Local path to model (optional)')
parser.add_argument('--total_tokens', type=int, default=100_000_000,
                    help='Total tokens to analyze')
parser.add_argument('--seq_len', type=int, default=128,
                    help='Sequence length for analysis')
parser.add_argument('--top_n', type=int, default=8,
                    help='Number of top activating sequences to save')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='Path to dataset for analysis')
parser.add_argument('--output_dir', type=str, default="./results",
                    help='Directory to save results')

args = parser.parse_args()

# Device setup
device = 'cuda'
dtype = torch.bfloat16

# Load base model
print("Loading base model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    args.model_path if args.model_path else args.model_name,
    torch_dtype=dtype,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path if args.model_path else args.model_name,
    use_fast=True,
    add_bos_token=True,
)

# Initialize TransformerLens wrapper
print("Initializing TransformerLens model...")
model = HookedTransformer.from_pretrained_no_processing(
    args.model_name,
    hf_model=hf_model,
    tokenizer=tokenizer,
    device=device,
    dtype=dtype,
)
model.eval()

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Load LoRSA model
print("Loading LoRSA model...")
lorsa = LowRankSparseAttention.from_pretrained(
    args.lorsa_path,
    device=device,
)
lorsa.fold_W_O_into_W_V()

# Define tokens to ignore in analysis
ignore_tokens = {
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
}

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(args.dataset_path)
# Select subset based on total tokens
dataset_subset = dataset.select(range(args.total_tokens // args.seq_len))

# Run analysis
print("Running analysis...")
sample_results = sample_max_activating_sequences(
    lorsa=lorsa, 
    dataset=dataset_subset, 
    model=model,
    ignore_tokens=ignore_tokens,
    batch_size=16,
    get_topn_activating_samples=args.top_n,
    seq_len=args.seq_len,
)

# Save results
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, 'sample_results.pt')
torch.save(sample_results, output_path)
print(f"Analysis complete! Results saved to {output_path}")