import os
import shutil
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

dataset_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/training_data/Slimpajama_B512_L256'

directory_list = []
for item in os.listdir(dataset_path):
    item_path = os.path.join(dataset_path, item)
    if os.path.isdir(item_path):
        directory_list.append(item_path)

file_paths = {
    'filter_mask': [],
    **({f'blocks.{l}.ln1.hook_normalized': [] for l in range(12)}),
    **({f'blocks.{l}.hook_attn_out': [] for l in range(12)}),
}

for directory_path in tqdm(directory_list, desc='Searching files'):
    for hook_name in file_paths.keys():
        hook_file_paths = []
        hook_file_dir = os.path.join(directory_path, hook_name)
        for item in os.listdir(hook_file_dir):
            item_path = os.path.join(hook_file_dir, item)
            if os.path.isfile(item_path) and item_path.endswith('.pt'):
                hook_file_paths.append(item_path)
        hook_file_paths.sort()
        file_paths[hook_name] += hook_file_paths
    
for key, value in file_paths.items():
    print(f"{key}: {len(value)} files")
    for i, file_path in enumerate(value[:3]):
        print(f"  {i}: {file_path}")
    print()

def move_files(file_list, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for i, file_path in enumerate(file_list):
        target_file_path = os.path.join(target_folder, f'{i}.pt')
        if os.path.exists(target_file_path):
            print(f"目标文件名已存在，无法移动: {target_file_path}")
            continue

        # 将文件从 temp_folder 移动到 target_folder
        shutil.move(file_path, target_file_path)

with ProcessPoolExecutor(max_workers=None) as executor:
    futures = [
        executor.submit(move_files, file_list, f'/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/training_data/Slimpajama_B512_L256/{key}')
        for key, file_list in file_paths.items()
    ]

    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"Error processing files: {e}")