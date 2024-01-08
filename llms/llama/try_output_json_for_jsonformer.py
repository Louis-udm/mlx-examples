# https://github.com/1rgs/jsonformer/tree/main

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Tuple, List
from sentencepiece import SentencePieceProcessor
import time
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from llama import load_model,toc

mx.random.seed(44)

"""
TinLlama >v0.2 uses a different prompt format
<|assistant|>
"""

# model_path="llama/TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted"
model_path="../../weights/llama/Llama-2-7b-chat-mlx"
if not Path(model_path).exists():
    model_path=model_path.replace("../","")

tokenizer_path="../../weights/llama/Llama-2-7b-chat-mlx/tokenizer.model"
if not Path(tokenizer_path).exists():
    tokenizer_path=tokenizer_path.replace("../","")

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.encode(*args, **kwds)
    
# tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
# tokenizer = Tokenizer(tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

model = load_model(model_path)

temp=0.8
max_tokens=1000
write_every=2


from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

prompt = "Generate a person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()

print(generated_data)