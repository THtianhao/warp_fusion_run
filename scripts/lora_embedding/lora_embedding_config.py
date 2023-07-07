from dataclasses import dataclass

@dataclass
class LoraEmbeddingConfig:
    lora_dir = '/content/drive/MyDrive/models/loras'  # @param {'type':'string'}
    custom_embed_dir = '/content/drive/MyDrive/models/embeddings'  # @param {'type':'string'}
