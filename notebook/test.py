import os

dataset_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/training_data/Slimpajama_B512_L256'

directories = []
for item in os.listdir(dataset_path):
    item_path = os.path.join(dataset_path, item)
    if os.path.isdir(item_path):
        directories.append(item_path)


pt_file_paths = []
for directory in directories:
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item_path.endswith('.pt'):
            pt_file_paths.append(item_path)

print(len(pt_file_paths))
print(pt_file_paths[0:10])
print(pt_file_paths[445:455])