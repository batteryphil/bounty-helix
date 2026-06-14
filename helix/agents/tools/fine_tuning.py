import torch
import torch
import numpy as np
import transformers
import torch.nn as nn



class LoRAConfig(transformers.LoraConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.supports_bias = False
        self.use_bias = False

    def forward(self, x):
        return x


class LoRATokenizer(transformers.LoraTokenizer):
    pass

class LoRA(model, model_type):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

