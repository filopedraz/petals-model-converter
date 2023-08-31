import torch
import json
import time
import logging
import os
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

model_name = "codellama/CodeLlama-34b-Instruct-hf"
output_folder = "codellama2"
number_of_hidden_layers = 48 # TODO get this info from `config.json`

start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache_dir")
logging.debug(f"Model loaded in Memory in {time.time() - start_time} seconds.\n")

dtype_of_weights = next(model.parameters()).dtype
logging.debug(f"Model weights are currently {dtype_of_weights}\n")

start_time = time.time()
model = model.to(torch.bfloat16)
logging.debug(f"Converted the weights to Brain Floating Point 16 in {time.time() - start_time} seconds.\n")

names = []
logging.debug("Model Structure")
for name, param in model.named_parameters():
    logging.debug(f"[[{name} {param.size()}]]")

names = list(name for name, _ in model.named_parameters())
logging.debug(f"The model contains {len(names)} tensors.")

weight_map = {}
for i in range(number_of_hidden_layers):
    for name in names:
        if name.startswith(f'model.layers.{i}.'):
            weight_map[name] = f"model_{i + 1:05}-of-000{number_of_hidden_layers+1}.safetensors"

for name in names:
    if name not in weight_map:
        weight_map[name] = f"model_000{number_of_hidden_layers+1}-of-000{number_of_hidden_layers+1}.safetensors"
logging.debug(f"Created weight map using {number_of_hidden_layers} blocks.")

total_size = sum(p.numel() * 2 for name, p in model.named_parameters())
index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
logging.debug(f"Total size in bytes of the model: {total_size}")

os.makedirs(output_folder, exist_ok=True)
logging.debug(f"Created folder to store the weights.")

with open(f"./{output_folder}/model.safetensors.index.json", "w") as f:
    f.write(json.dumps(index, indent=4) + "\n")
logging.debug(f"Created model config file for the resharded model.")

for shard in sorted(set(weight_map.values())):
    tensors = {name: p for name, p in model.named_parameters() if weight_map[name] == shard}
    logging.debug(f"Saving {sorted(tensors.keys())} to {shard}")
    save_file(tensors, f"./{output_folder}/" + shard, metadata={"format": "pt"})