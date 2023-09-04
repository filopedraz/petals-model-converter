# Petals Model Converter

Reshard any HF Model for Petals Decentralized Inference Engine.

## Getting Started

### 1. Change the variables

```python
model_name = "codellama/CodeLlama-34b-Instruct-hf"
output_folder = "models/CodeLlama-34b-Instruct-hf"
number_of_hidden_layers = 48 # You can get this info in the config.json file of each HF repository
```

### 2. Run the script

```python
python main.py
```

## 3. Copy Files

When the conversion has been completed, copy the following files from the official model Repository and push the new model in HF.

- `.gitattributes`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `tokenizer.model`