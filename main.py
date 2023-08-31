import torch
import json

from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

model_name = "codellama/CodeLlama-34b-Instruct-hf"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache_dir")
print("Model loaded in Memory.")

model = model.to(torch.bfloat16)

names = list(name for name, _ in model.named_parameters())
print(names)

exit()

weight_map = {}
for i in range(80):
    for name in names:
        if name.startswith(f'model.layers.{i}.'):
            weight_map[name] = f"model_{i + 1:05}-of-00081.safetensors"

for name in names:
    if name not in weight_map:
        weight_map[name] = "model_00081-of-00081.safetensors"

total_size = sum(p.numel() * 2 for name, p in model.named_parameters())
index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}


with open('./cache_dir/beluga/model.safetensors.index.json', 'w') as f:
    f.write(json.dumps(index, indent=4) + '\n')

for shard in sorted(set(weight_map.values())):
    tensors = {name: p for name, p in model.named_parameters.items() if weight_map[name] == shard}
    print(f"Saving {sorted(tensors.keys())} to {shard}")
    save_file(tensors, "./cache_dir/beluga/" + shard, metadata={"format": "pt"})