import torch
import json
import logging
import os

from safetensors.torch import save_file

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

input_folder = "models/example/CodeLlama-34b-Instruct-hf"
output_folder = "models/CodeLlama-34b-Instruct-hf"

number_of_hidden_layers = 48
number_of_current_shards = 7

os.makedirs(output_folder, exist_ok=True)

total_size = 0
weight_map = {}
for k in range(number_of_hidden_layers):
    shard_name = f"model_000{k+1}-of-000{number_of_hidden_layers+1}.safetensors"

    for i in range(1, number_of_current_shards + 1):
        chunk = torch.load(f"{input_folder}/pytorch_model-0000{i}-of-0000{number_of_current_shards}.bin", map_location="cpu")

        shard_tensors = {}
        for name, tensor in chunk.items():
            if name.startswith(f'model.layers.{k}.'):
                converted_tensor = tensor.to(torch.bfloat16)
                total_size += converted_tensor.numel() * 2
            
                weight_map[name] = shard_name
                shard_tensors[name] = converted_tensor
        
        if len(shard_tensors.keys()) > 0:
            save_file(shard_tensors, f"./{output_folder}/{shard_name}", metadata={"format": "pt"})
            logging.debug(f"Saved {' '.join(shard_tensors.keys())} in shard {shard_name}.")
        
        del chunk
        del shard_tensors

shard_name = f"model_000{number_of_hidden_layers+1}-of-000{number_of_hidden_layers+1}.safetensors"
for i in range(1, number_of_current_shards + 1):
    chunk = torch.load(f"{input_folder}/pytorch_model-0000{i}-of-0000{number_of_current_shards}.bin", map_location="cpu")

    shard_tensors = {}
    for name, tensor in chunk.items():
        if name not in weight_map:
            converted_tensor = tensor.to(torch.bfloat16)
            total_size += converted_tensor.numel() * 2
        
            weight_map[name] = shard_name
            shard_tensors[name] = converted_tensor
    
    if len(shard_tensors.keys()) > 0:
        save_file(shard_tensors, f"./{output_folder}/{shard_name}", metadata={"format": "pt"})
        logging.debug(f"Saved {' '.join(shard_tensors.keys())} in shard {shard_name}.")
    
    del chunk
    del shard_tensors

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
with open(f"./{output_folder}/model.safetensors.index.json", "w") as f:
    f.write(json.dumps(index, indent=4) + "\n")
logging.debug("Created model config file for the resharded model.")