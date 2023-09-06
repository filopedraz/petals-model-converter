import torch
import json
import logging
import requests
import os
import io
from safetensors.torch import save_file

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

model_name = "codellama/CodeLlama-34b-Instruct-hf"
output_folder = "models/CodeLlama-34b-Instruct-hf"
number_of_hidden_layers = 48
number_of_current_shards = 7

weight_map = {}
shard_tensors = {f"model_{i + 1:05}-of-000{number_of_hidden_layers+1}.safetensors": {} for i in range(number_of_hidden_layers)}
shard_tensors[f"model_000{number_of_hidden_layers+1}-of-000{number_of_hidden_layers+1}.safetensors"] = {}

for i in range(1, number_of_current_shards + 1):
    chunk_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model-0000{i}-of-0000{number_of_current_shards}.bin"
    
    response = requests.get(chunk_url)
    response.raise_for_status()
    
    buffer = io.BytesIO(response.content)
    chunk = torch.load(buffer, map_location="cpu")

    for name, tensor in chunk.items():
        if name.startswith('model.layers.') and any(str(i) in name for i in range(number_of_hidden_layers)):
            shard_name = f"model_{int(name.split('.')[2]) + 1:05}-of-000{number_of_hidden_layers+1}.safetensors"
        else:
            shard_name = f"model_000{number_of_hidden_layers+1}-of-000{number_of_hidden_layers+1}.safetensors"
        
        shard_tensors[shard_name][name] = tensor.to(torch.bfloat16)
        
        weight_map[name] = shard_name
        del tensor

    del chunk
    logging.debug(f"Processed shard {i} out of {number_of_current_shards}.")

total_size = sum(p.numel() * 2 for shard in shard_tensors.values() for p in shard.values())
index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
logging.debug(f"Total size in bytes of the model: {total_size}")

os.makedirs(output_folder, exist_ok=True)
logging.debug("Created folder to store the weights.")

with open(f"./{output_folder}/model.safetensors.index.json", "w") as f:
    f.write(json.dumps(index, indent=4) + "\n")
logging.debug("Created model config file for the resharded model.")

for shard, tensors in shard_tensors.items():
    if tensors:
        logging.debug(f"Saving tensors to {shard}")
        save_file(tensors, f"./{output_folder}/{shard}", metadata={"format": "pt"})
        del tensors

logging.info("Processing completed.")
