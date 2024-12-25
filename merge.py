import os
import shutil
from tqdm import tqdm
# Paths
base_dir = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Language-Model-SAEs/crosscoder_act_shuffled_context"
output_dir = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Language-Model-SAEs/merged_acts"  # Where the final blocks.Y directories will be created

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for split_dir in tqdm(os.listdir(base_dir)):
    if split_dir.startswith("split_") and os.path.isdir(os.path.join(base_dir, split_dir)):
        split_number = split_dir.split("_")[1]  # Extract the number from "split_X"
        split_path = os.path.join(base_dir, split_dir)

        for block_dir in os.listdir(split_path):
            block_path = os.path.join(split_path, block_dir)

            if block_dir.startswith("blocks.") and os.path.isdir(block_path):  # 这里往下的逻辑要改一下
                for file_name in os.listdir(block_path):
                    file_path = os.path.join(block_path, file_name)

                    if os.path.isfile(file_path) and file_name.startswith("shard-0"):
                        # Construct new file name
                        new_file_name = file_name.replace("shard-0", f"shard-{split_number}", 1)
                        new_file_path = os.path.join(block_path, new_file_name)

                        # Rename the file
                        os.rename(file_path, new_file_path)

# Loop over split_X directories
for split_dir in tqdm(os.listdir(base_dir)):
    split_path = os.path.join(base_dir, split_dir)

    if split_dir.startswith("split_") and os.path.isdir(split_path):
        for block_dir in os.listdir(split_path):
            block_path = os.path.join(split_path, block_dir)

            if block_dir.startswith("blocks.") and os.path.isdir(block_path):
                # Create the corresponding blocks.Y directory in the output folder
                target_block_path = os.path.join(output_dir, block_dir)
                os.makedirs(target_block_path, exist_ok=True)

                # Move all .pt files from this blocks.Y to the target blocks.Y
                for file_name in os.listdir(block_path):
                    file_path = os.path.join(block_path, file_name)

                    if os.path.isfile(file_path) and file_name.endswith(".pt"):
                        target_file_path = os.path.join(target_block_path, file_name)
                        shutil.move(file_path, target_file_path)


shutil.rmtree(base_dir)
os.rename(output_dir, base_dir)


