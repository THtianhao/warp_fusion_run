# @title LORA & embedding paths
import os
import torch
from scripts.lora_embedding.lora_embedding_config import LoraEmbeddingConfig
from scripts.lora_embedding.lora_embedding_fun import list_available_loras, available_loras

def set_lora_embedding(config: LoraEmbeddingConfig):
    if config.lora_dir.startswith('/content'):
        config.lora_dir = './loras'
        print('Overriding lora dir to ./loras for non-colab env because you path begins with /content. Change path to desired folder')

    if config.custom_embed_dir.startswith('/content'):
        custom_embed_dir = './embeddings'
        os.makedirs(custom_embed_dir, exist_ok=True)
        print('Overriding embeddings dir to ./embeddings for non-colab env because you path begins with /content. Change path to desired folder')

    if not hasattr(torch.nn, 'Linear_forward_before_lora'):
        torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

    if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
        torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

    if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

    if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
        torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

    if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
        torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

    if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
        torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

    list_available_loras(config.lora_dir)
    print('Loras detected:\n', '\n'.join(list(available_loras.keys())))
